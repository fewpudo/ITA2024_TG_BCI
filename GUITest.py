import customtkinter as ctk

class BCIGUI(ctk.CTk):
    def __init__(self, on_confirm):
        super().__init__()

        self.title("BCI GUI")
        self.geometry("800x600")
        self.on_confirm = on_confirm  # Função a ser chamada ao confirmar

        # Rotina de Treinamento
        self.label_training = ctk.CTkLabel(self, text="Rotina de Treinamento:")
        self.label_training.pack(pady=10)

        self.training_var = ctk.BooleanVar()
        self.training_switch = ctk.CTkSwitch(self, text="Ativar Treinamento", variable=self.training_var)
        self.training_switch.pack(pady=10)

        # Frequências Desejadas
        self.label_frequencies = ctk.CTkLabel(self, text="Frequências Desejadas (Hz):")
        self.label_frequencies.pack(pady=10)

        self.freq_entry = ctk.CTkEntry(self)
        self.freq_entry.pack(pady=10)

        # Seleção de Canais
        self.label_channels = ctk.CTkLabel(self, text="Seleção de Canais:")
        self.label_channels.pack(pady=10)

        self.channels = ["C3", "C4", "Pz", "Oz", "F3", "F4"]
        self.selected_channels = []

        for channel in self.channels:
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(self, text=channel, variable=var)
            chk.pack(anchor="w")
            self.selected_channels.append((channel, var))

        # Botão de Confirmação
        self.confirm_button = ctk.CTkButton(self, text="Confirmar", command=self.confirm_selection)
        self.confirm_button.pack(pady=20)

    def confirm_selection(self):
        is_training = self.training_var.get()
        frequencies = self.freq_entry.get()
        selected_channels = [channel for channel, var in self.selected_channels if var.get()]

        # Chama a função passada e envia os dados
        self.on_confirm(is_training, frequencies, selected_channels)

if __name__ == "__main__":
    app = BCIGUI(lambda x, y, z: print(f"Treinamento: {x}, Frequências: {y}, Canais: {z}"))
    app.mainloop()
