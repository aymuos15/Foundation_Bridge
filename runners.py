import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_lr_lambda(initial_lr=0.001):
    def lr_lambda(epoch):
        if epoch < 50:
            return 1
        elif epoch < 75:
            return 0.1
        else:
            return 0.01
    return lr_lambda

def train_model(model, train_loader, criterion, optimizer, scheduler, task, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).all(dim=1).sum().item()
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        total += targets.size(0)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    scheduler.step()
    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy

def evaluate_model(model, test_loader, criterion, task, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            total += targets.size(0)

    accuracy = 100. * correct / total
    return accuracy

def train_test(model, train_loader, test_loader, task, epochs):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, get_lr_lambda())

    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, scheduler, task, DEVICE)

    test_acc = evaluate_model(model, test_loader, criterion, task, DEVICE)
    return test_acc, model

def train_test_private(model, train_loader, test_loader, task, epochs):
    model = ModuleValidator.fix(model).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=1.2,
        target_delta=1 / len(train_loader.dataset),
        max_grad_norm=1.2,
        poisson_sampling=False
    )

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=64,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for epoch in range(epochs):
            train_model(model, memory_safe_data_loader, criterion, optimizer, scheduler, task, DEVICE)

    test_acc = evaluate_model(model, test_loader, criterion, task, DEVICE)
    return test_acc, model