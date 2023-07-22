from torch.nn.modules.module import Module
import torch.nn as nn

class add(nn.Module):
    def __init__(self):
        super(add, self).__init__()
        
    def forward(self, adj, A2, A3, A4, A5):
        AST_output =0.7*adj +0.2*A2+0.1*A3
        # AST_output = self.ffn(AST_output)

        return AST_output
        