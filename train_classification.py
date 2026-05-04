import argparse

import torch
import torch.optim as optim
from torch import nn
import torchio as tio

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics
import nibabel
import numpy as np

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot

from models.resnet import resnet50
from models.simple import CT3DClassifier

def test_plot_matrix(volume, name):
    # Threshold to show only relevant structures
    threshold = volume.mean()
    voxels = volume > threshold

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(voxels, edgecolor='k')
    matplotlib.pyplot.savefig(name)
    matplotlib.pyplot.close()

def get_transform():
    return tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=30,
        translation=10,
        isotropic=True,
        image_interpolation='linear'
    )

def train_model(model, train_loader, val_loader, num_epochs, learning_rate=1e-4, device='cuda', exp_name='default'):
    """
    Bucle de entrenamiento para clasificación 3D.
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([414 / (2* 400), 414 / (2 * 1)]).to(device))
#    pos_weight = torch.tensor([num_neg, num_pos])
#    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)#, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")

        model.train() 
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc="Entrenando"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1) # Obtiene la clase con mayor probabilidad
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # =========================
        #    FASE DE VALIDACIÓN
        # =========================
        model.eval() # Desactiva Dropout, fija BatchNorm
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validando "):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                #_, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # prob clase positiva
                predicted = (probs > 0.1).int()  # threshold ajustable
                # print(outputs)
                # print(probs)
                # print(predicted)
                # print(labels)
                # print("Loss", loss)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.append(predicted.cpu())
                all_targets.append(labels.cpu())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        epoch_val_f2 = sklearn.metrics.fbeta_score(all_targets, all_preds, beta=2)
        cm = sklearn.metrics.confusion_matrix(all_targets, all_preds)

        # Actualizar el scheduler con la pérdida de validación
        scheduler.step(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        print(f"Val F2-score: {epoch_val_f2}")
        print(cm)

        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"mejor_modelo_3d{exp_name}.pth")
            print(">>> ¡Nuevo mejor modelo guardado!")

    print("\nEntrenamiento finalizado.")
    return model


def load_medicalnet_weights(model, weight_path):
    """
    Carga los pesos preentrenados filtrando las capas que no coinciden 
    (por ejemplo, la nueva capa final fc).
    """
    print(f"Cargando pesos desde: {weight_path}")
    
    # Cargar el archivo .pth
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Los modelos guardados a veces envuelven los pesos en 'state_dict'
    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint
        
    # Eliminar prefijos extra si el modelo original se entrenó con DataParallel (p. ej. 'module.')
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Obtener el diccionario del modelo actual (clasificador)
    model_dict = model.state_dict()

    # Filtrar: solo nos quedamos con los pesos que existen en nuestro modelo Y tienen el mismo tamaño
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }

    # Actualizar nuestro modelo con los pesos filtrados
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    print(f"Pesos cargados con éxito. Se restauraron {len(filtered_dict)} de {len(model_dict)} capas de pesos.")
    return model

def load_file_list(txt_path, label):
    """Read a file with a file list and generates an array with label for each file"""
    with open(txt_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    labels = [label] * len(files)
    return files, labels

class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, files, labels, transform=None):
        self.data_path = data_path
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load NIfTI
        img = nibabel.load(self.data_path + self.files[idx])
        data = img.get_fdata().astype(np.float32)

        # torchio expects shape (C, D, H, W)
        data = torch.from_numpy(data).unsqueeze(0)
        # test_plot_matrix(data.squeeze(), name=f"/tmp/Test{idx}-orig.png")
        if self.transform:
            subject = tio.Subject(img=tio.ScalarImage(tensor=data))
            subject = self.transform(subject)
            data = subject.img.data

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # print(f"{idx} Data shape: {data.shape}")
        # test_plot_matrix(data.squeeze(), name=f"/tmp/Test{idx}.png")
        return data, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Learning classifier for 3D images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('path', help='Path to Nifti images (ie. data/cuerpoV/)')
    parser.add_argument('negative', help='File (.txt) with a list of files for negative class (ie. "sin_colapso-filtered.txt")')
    parser.add_argument('positive', help='File (.txt) with a list of files for positive class (ie. "colapsadas.txt")')
    parser.add_argument('-m', '--model', default='resnet50', help='Model to train (ie. simple)')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('-f', '--frozen', action='store_true', help='If resnet50 whether to freeze the backbone or not')
    args = parser.parse_args()


    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 0. Preparar carga de datos
    data_path = args.path
    files_normal,   labels_normal   = load_file_list(data_path + args.negative, 0)
    files_collapse, labels_collapse = load_file_list(data_path + args.positive, 1)
    all_files  = files_normal  + files_collapse
    all_labels = labels_normal + labels_collapse

    train_files, val_files, train_labels, val_labels = train_test_split(all_files, all_labels,
                                        test_size=0.2, random_state=42, stratify=all_labels)

    train_dataset = NiftiDataset(data_path, train_files, train_labels, transform=get_transform())
    val_dataset   = NiftiDataset(data_path, val_files, val_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 1. Instanciar el modelo 
    num_classes = 2
    if args.model == 'resnet50':  # (resnet50 modificado para 2 clases)
        model = resnet50(num_classes=num_classes) 
        model_path = "pretrain/resnet_50.pth"

        # 2. Cargar los pesos
        model = load_medicalnet_weights(model, model_path)

        # 3. Congelar el backbone de MedicalNet
        if args.frozen: 
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif args.model == 'simple':
        model = CT3DClassifier()
    else:
        print(f'Unknown model: {args.model}')


    # 4. Iniciar el entrenamiento (suponiendo que train_loader y val_loader están creados)
    model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=1e-4, device=device, 
                        exp_name=f"{args.model}{args.path.rstrip('/').split('/')[-1]}{args.epochs}{'frozen' if args.frozen else ''}")
    