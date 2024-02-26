import torch
import torch.nn as nn
import ipdb


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, con_encoder_layer_sizes, con_latent_size):

        super().__init__()


        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, con_latent_size)

        self.con_encoder = ConEncoder(
            con_encoder_layer_sizes, con_latent_size)

        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, con_latent_size)

    def forward(self, x, c=None):

        c = self.con_encoder(c)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        c = self.con_encoder(c)
        recon_x = self.decoder(z, c)

        return recon_x


class ConEncoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, order=False):

        super().__init__()

        self.order = order

        
        self.MLP_history = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_history.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP_history.add_module(name="A{:d}".format(i), module=nn.ReLU())


        self.MLP_env = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_env.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP_env.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP_neighbor = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_neighbor.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP_neighbor.add_module(name="A{:d}".format(i), module=nn.ReLU())


        
        if self.order == True:
            self.MLP_neighbor1 = nn.Sequential()

            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP_neighbor1.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                self.MLP_neighbor1.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP_condiction = nn.Sequential()
        self.MLP_condiction.add_module(
            name="L{:d}".format(0), module=nn.Linear(4*layer_sizes[-1], latent_size))
        self.MLP_condiction.add_module(name="A{:d}".format(0), module=nn.ReLU())
        

    def forward(self, c):

        if self.order == True:
            c0 = self.MLP_history(c[:,0,:])
            c1 = self.MLP_neighbor(c[:,1,:])
            c2 = self.MLP_neighbor1(c[:,2,:])
            c3 = self.MLP_env(c[:,3,:])
        else:
            c0 = self.MLP_history(c[:,0,:])
            c1 = self.MLP_neighbor(c[:,1,:])
            c2 = self.MLP_neighbor(c[:,2,:])
            c3 = self.MLP_env(c[:,3,:])

        c = torch.cat([c0, c1, c2, c3],dim=1)
        y = self.MLP_condiction(c)
    
        return y


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, condiction_size):

        super().__init__()


        layer_sizes[0] += condiction_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)


    def forward(self, x, c):



        x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, condiction_size):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size + condiction_size


        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())


    def forward(self, z, c):


        z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
