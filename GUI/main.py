import customtkinter as ctk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Desafios no momento => Como passar os dados selecionados na GUI para a função de aquisição?
# Alterar a função de aquisição para considerar os dados vindos da OpenBCI
# Melhorar a interface gráfica para exibir os dados em tempo real
# OpenBCI são 16 canais somente.

class BCI_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BCI GUI")
        self.geometry("800x600")

        self.channels = ["01", "02", "0z"]
        self.selected_channels = []
        self.problematic_channels = []
        self.sampling_rate = 250  # Hz

        self.create_widgets()

    def create_widgets(self):

        self.label_channels = ctk.CTkLabel(self, text="Selecione os Canais:", font=("Roboto", 16))
        self.label_channels.pack(pady=12, padx=10)

        self.channel_vars = []
        for channel in self.channels:
            var = tk.IntVar()
            chk = ctk.CTkCheckBox(self, text=channel, variable=var)
            chk.pack(anchor="center")
            self.channel_vars.append(var)

        print(self.channel_vars[0])
        
        #Selecionar tempo de aquisição -> Descobrir como passar pra dentro da função de forma eficiente
        self.label_time = ctk.CTkLabel(self, text="Tempo de Aquisição (segundos):", font=("Roboto", 16))
        self.label_time.pack(pady=12, padx=10)

        self.entry_time = ctk.CTkEntry(self)
        self.entry_time.pack(pady=12, padx=10)

        # Iniciar aquisição => Parte da BCI online! 
        self.start_button = ctk.CTkButton(self, text="Iniciar Aquisição", command=self.start_acquisition) #chama a função de aquisição
        self.start_button.pack(pady=20)

        # Melhorar parte do Plot
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(pady=20)


    def start_acquisition(self):
       print("Aquisição iniciada", self.channel_vars, self.selected_channels)
   

if __name__ == "__main__":
    app = BCI_GUI()
    app.mainloop()