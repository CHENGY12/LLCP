import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb


class LSTMVAE(nn.Module):

    def __init__(self, input_size, latent_size, output_size):
        super(LSTMVAE, self).__init__()

        self.num_layers = 1
        self.latent_size = latent_size

        self.lstm_encoder = nn.LSTM(input_size=input_size * 10, hidden_size=latent_size,
                                    num_layers=1, batch_first=True).cuda()
        self.lstm_decoder = nn.LSTM(input_size=latent_size, hidden_size=output_size,
                                    num_layers=1, batch_first=True).cuda()

        self.linear_mean = nn.Linear(latent_size, latent_size).cuda()
        self.linear_log_var = nn.Linear(latent_size, latent_size).cuda()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        #  shape of x: batch_size * 10, 16, 1
        batch_size = x.size(0)

        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.latent_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.latent_size)).cuda()
        x = torch.tile(x, [1, 1, 10])
        pos, (h_out, _) = self.lstm_encoder(x, (h_0, c_0))  # shape of pos: batch_size*10, seq_len, hidden_size
        pos = self.relu(pos)
        means = self.linear_mean(pos)
        log_var = self.linear_log_var(pos)

        z = self.reparameterize(means, log_var) # shape of z: batch_size * 10,  seq_len, hidden_size

        dec_h_0 = Variable(torch.zeros(self.num_layers, batch_size, 1)).cuda()
        dec_c_0 = Variable(torch.zeros(self.num_layers, batch_size, 1)).cuda()

        dec_pos, (dec_h_out, _) = self.lstm_decoder(z, (dec_h_0, dec_c_0))
        dec_pos = self.sigmoid(dec_pos)
        #   shape of dec_pos: batch_size*10, seq_len, 1
        return dec_pos, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z):
        #  the shape of z: batch_size * 10,  seq_len, hidden_size
        batch_size = z.size(0)
        dec_h_0 = Variable(torch.zeros(self.num_layers, batch_size, 1)).cuda()
        dec_c_0 = Variable(torch.zeros(self.num_layers, batch_size, 1)).cuda()

        dec_pos, (dec_h_out, _) = self.lstm_decoder(z, (dec_h_0, dec_c_0))

        return dec_pos
        pass


class SimpleVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, con_encoder_layer_sizes, con_latent_size):
        super(SimpleVAE, self).__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

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

        # shape of c : B * 4 * d
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


class SimpleEncoder(nn.Module):

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
