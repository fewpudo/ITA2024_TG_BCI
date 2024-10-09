import customtkinter as ctk

import acquisition

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

        # Tempo de treinamento
        self.label_trainingTime = ctk.CTkLabel(self, text="Tempo de Treinamento (s):")
        self.label_trainingTime.pack(pady=10)

        self.trainningTime_entry = ctk.CTkEntry(self)
        self.trainningTime_entry.pack(pady=10)

        # Canais
        self.label_channels = ctk.CTkLabel(self, text="Canais:")
        self.label_channels.pack(pady=10)

        self.channels_entry = ctk.CTkEntry(self)
        self.channels_entry.pack(pady=10)

        # Trials
        self.label_trials = ctk.CTkLabel(self, text="Trials:")
        self.label_trials.pack(pady=10)

        self.trials_entry = ctk.CTkEntry(self)
        self.trials_entry.pack(pady=10)

        # Botão de Confirmação
        self.confirm_button = ctk.CTkButton(self, text="Confirmar", command=self.confirm_selection)
        self.confirm_button.pack(pady=20)

    def confirm_selection(self):
        is_training = self.training_var.get()
        frequencies = int(self.freq_entry.get())
        trainningTime = int(self.trainningTime_entry.get())
        channels = int(self.channels_entry.get())
        trials = int(self.trials_entry.get())

        if is_training:
            w = acquisition.trainningAcquisition(trainningTime, channels, frequencies, trials, is_training)
        else:
            y_pred = acquisition.BCIOnline(trainningTime, w)
            print(y_pred)
        # Chama a função passada e envia os dados
        self.on_confirm(is_training, frequencies, trainningTime, channels, trials, w)

if __name__ == "__main__":
    app = BCIGUI(lambda x, y, z: print(f"Treinamento: {x}, Frequências: {y}, Canais: {z}"))
    app.mainloop()

# Como salvar o classificador para usar na parte online?