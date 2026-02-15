import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from .dataset import ShadowDataset
from .model import CholeskyTomographyNet
from .utils import quantum_fidelity, trace_distance
import time

def train_model(num_samples=1000, epochs=10, batch_size=32):
    # 1. Prepare Data
    dataset = ShadowDataset(num_samples=num_samples, num_qubits=1, num_measurements_per_state=100)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CholeskyTomographyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training on {device}...")
    
    # 3. Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (measurements, rho_target) in enumerate(train_loader):
            measurements = measurements.to(device)
            rho_target = rho_target.to(device)
            
            optimizer.zero_grad()
            
            rho_pred = model(measurements)
            
            # Loss: 1 - Fidelity
            # Compute fidelity for batch
            # Fidelity(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2
            # For 1 qubit, analytical formula or trace distance might be easier for gradient?
            # Let's use MSE on density matrix elements for stability during training + constraints are enforced by architecture
            # MSE Loss = mean(|rho_pred - rho_target|^2)
            
            loss = torch.mean(torch.abs(rho_pred - rho_target)**2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # 4. Save Model
    torch.save(model.state_dict(), 'outputs/model.pth')
    
    # 5. Evaluation
    model.eval()
    fidelities = []
    trace_dists = []
    latencies = []
    
    with torch.no_grad():
        for measurements, rho_target in test_loader:
            measurements = measurements.to(device)
            
            t0 = time.time()
            rho_pred = model(measurements)
            t1 = time.time()
            latencies.append((t1 - t0) / measurements.shape[0])
            
            # Metrics (CPU side)
            r_pred_np = rho_pred.cpu().numpy()
            r_true_np = rho_target.numpy()
            
            for i in range(len(r_pred_np)):
                f = quantum_fidelity(r_pred_np[i], r_true_np[i])
                td = trace_distance(r_pred_np[i], r_true_np[i])
                fidelities.append(f)
                trace_dists.append(td)
                
    mean_fidelity = np.mean(fidelities)
    mean_trace = np.mean(trace_dists)
    avg_latency = np.mean(latencies)
    
    print("\n=== Final Results ===")
    print(f"Mean Fidelity: {mean_fidelity:.4f}")
    print(f"Mean Trace Distance: {mean_trace:.4f}")
    print(f"Inference Latency: {avg_latency*1000:.4f} ms/state")
    
    return mean_fidelity, mean_trace

if __name__ == "__main__":
    train_model()
