import torch
import torch.nn as nn

class CholeskyTomographyNet(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2, dim_rho=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, embed_dim)) # Max M=500
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: Predict L matrix (Lower Triangular)
        # L has dim_rho * (dim_rho + 1) / 2 complex parameters?
        # Actually simpler: Predict full matrix then mask? 
        # Or predict dim_rho^2 complex numbers and enforce structure.
        # Let's predict flattened real and imag parts for full matrix then extract lower triangle.
        # For 2x2:
        # L = [[l00, 0], [l10, l11]]
        # l00, l11 must be real/positive for uniqueness? 
        # For general CPTP/PSD, L can be any lower triangular Complex.
        # But diagonal elements must be real and positive for Cholesky usually?
        # Actually rho = L L^dag. L just needs to be lower triangular.
        # Elements: l00 (real, >0 for uniqueness, but we just need ANY L), l10 (complex), l11 (real >0)
        # Total parameters:
        # 2x2: 
        # L = [[a, 0], [b+ic, d]]
        # 4 real params: a, b, c, d.
        
        self.head = nn.Linear(embed_dim, 4) # Hardcoded for 1 qubit (2x2 rho)
        
    def forward(self, x):
        # x: (Batch, Seq_Len)
        b, s = x.shape
        
        # Embedding
        emb = self.embedding(x) + self.pos_encoder[:, :s, :]
        
        # Transformer
        out = self.transformer(emb)
        
        # Pool (mean over sequence)
        out = out.mean(dim=1)
        
        # Predict L parameters
        params = self.head(out)
        
        # Construct L
        # params: [a, b, c, d]
        # L = [[a, 0], [b+ic, d]]
        
        L_real = torch.zeros(b, 2, 2, device=x.device)
        L_imag = torch.zeros(b, 2, 2, device=x.device)
        
        a = params[:, 0]
        b_val = params[:, 1]
        c = params[:, 2]
        d_val = params[:, 3]
        
        # Fill L (Lower Triangular)
        # Row 0
        L_real[:, 0, 0] = a 
        
        # Row 1
        L_real[:, 1, 0] = b_val
        L_imag[:, 1, 0] = c
        L_real[:, 1, 1] = d_val
        
        L = torch.complex(L_real, L_imag)
        
        # Reconstruct Rho = L @ L^dag
        L_dag = torch.transpose(L.conj(), 1, 2)
        rho_raw = torch.matmul(L, L_dag)
        
        # Normalization: Tr(rho) = 1
        trace = torch.einsum('bii->b', rho_raw).real # Trace is real for Hermitian
        trace = trace.view(-1, 1, 1) + 1e-8 # Avoid div by zero
        
        rho = rho_raw / trace
        
        return rho
