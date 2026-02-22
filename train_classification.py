import torch
from models.resnet import resnet50
import torch.optim as optim
from torch import nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, learning_rate=1e-4, device='cuda'):
    """
    Bucle de entrenamiento para clasificación 3D.
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

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
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validando "):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        # Actualizar el scheduler con la pérdida de validación
        scheduler.step(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        
        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "mejor_modelo_3d.pth")
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Ejemplo de uso ---
    # 1. Instanciar el modelo (por ejemplo, resnet50 modificado para 2 clases)
    num_classes = 2
    model = resnet50(num_classes=num_classes) 
    print(model)
    model_path = "path/to/models"

    # 2. Cargar los pesos
    model = load_medicalnet_weights(model, model_path)


    # 3. (Opcional) Congelar el backbone 
    # for name, param in model.named_parameters():
    #     if "classifier" not in name:
    #         param.requires_grad = False

    # 4. Iniciar el entrenamiento (suponiendo que train_loader y val_loader están creados)
    # model = train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4, device=device)