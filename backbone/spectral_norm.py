import torch
from torch.nn import Parameter

##Memastikan bahwa panjang vektor tidak lebih besar dari 1
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

#Menerapkan normalisasi spektral pada neural network weights (Parameter yang digunakan oleh neural network untuk mempelajari pola dalam data). 
#Mengontrol bobot selama training network untuk mencegah eksplosinya.
class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        #print(self.name)
        self.power_iterations = 1
    #Menjaga agar distribusi bobot lapisan neural network tetap stabil dengan membatasi bobot berdasarkan nilai singularnya.
    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    # Fungsi statis yang digunakan untuk menerapkan Spectral Normalization pada modul atau lapisan tertentu di jaringan neural.
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)

            #del module._parameters[name]

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    #Menghapus normalisasi spektral dari modul yang telah dipasangi Spectral Normalization.
    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

#Menambahkan Spectral Normalization pada lapisan jaringan neural seperti lapisan konvolusi atau lapisan linier.
def spectral_norm(module):
    SpectralNorm.apply(module)
    return module

#Digunakan untuk menghapus Spectral Normalization jika tidak diperlukan lagi selama proses training maupun testing.
def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))

#Spectral Normalization mengontrol bobot neural network dengan membatasi nilai singular terbesar dari matriks bobot. 
#Hal ini sangat penting dalam model seperti GAN yang cenderung tidak stabil,
#karena bobot yang tidak terkontrol dapat menyebabkan mode collapse atau pelatihan yang tidak seimbang.