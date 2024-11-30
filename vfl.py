import torch

class Client(torch.nn.Module):
    def __init__(self, client_model):
        super(Client, self).__init__()
        self.client_model = client_model
        self.client_side_intermediate = None
        self.grad_from_server = None


    def forward(self, inputs):
        self.client_side_intermediate = self.client_model(inputs)
        # intermediate_to_server = self.client_side_intermediate.detach()\
        #     .requires_grad_()
        intermediate_to_server = self.client_side_intermediate
        # client的中间结果传给Server
        return intermediate_to_server
    
    def client_backward(self, grad_from_server):
        """
        This function is used to backpropagate the gradients from the server to the client model
        """
        self.grad_from_server = grad_from_server
        self.client_side_intermediate.backward(grad_from_server)

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()


class Server(torch.nn.Module):
    def __init__(self, server_model, cat_dimension):
        super(Server, self).__init__()
        self.server_model = server_model
        self.cat_dimension = cat_dimension
        self.intermediate_to_server = [] # List of intermediate values from the clients
        self.grad_to_client = []
        self.input = None

    def forward(self, intermediate_to_server):
        self.intermediate_to_server = intermediate_to_server
        # Concatenate the intermediate values from the clients
        input = torch.cat((self.intermediate_to_server[0], self.intermediate_to_server[1]), self.cat_dimension)
        self.input = input
        output = self.server_model(input)
        return output
    
    def server_backward(self):
        """
        This function is used to backpropagate the gradients from the server to the clients
        """
        self.grad_to_client = [self.intermediate_to_server[0].grad.clone(), self.intermediate_to_server[1].grad.clone()]
        return self.grad_to_client

    def train(self):
        self.server_model.train()


    def eval(self):
        self.server_model.eval()

    

class VFLNN(torch.nn.Module):
    def __init__(self, client1, client2, server, client_optimizer, server_optimizer):
        super(VFLNN, self).__init__()
        self.client1 = client1 # client1是 fb
        self.client2 = client2 # client2是 fa
        self.server = server

        self.client1_optimizer = client_optimizer[0]
        self.client2_optimizer = client_optimizer[1]
        self.server_optimizer = server_optimizer
        self.intermediate_to_server1 = None
        self.intermediate_to_server2 = None

    def forward(self, inputs1, inputs2):
        # Client的中间结果传给Server
        self.intermediate_to_server1 = self.client1(inputs1).detach().requires_grad_()
        self.intermediate_to_server2 = self.client2(inputs2).detach().requires_grad_()
        
        output = self.server([self.intermediate_to_server1, self.intermediate_to_server2])
    
        return output

    def backward(self):
        grad_to_client = self.server.server_backward()
        # Backpropagate the gradients to the clients
        self.client1.client_backward(grad_to_client[0])
        self.client2.client_backward(grad_to_client[1])

    def zero_grads(self):
        self.client1_optimizer.zero_grad()
        self.client2_optimizer.zero_grad()
        self.server_optimizer.zero_grad()

    def step(self):
        self.client1_optimizer.step()
        self.client2_optimizer.step()
        self.server_optimizer.step()

    def train(self):
        self.client1.train()
        self.client2.train()
        self.server.train()

    def eval(self):
        self.client1.eval()
        self.client2.eval()
        self.server.eval()