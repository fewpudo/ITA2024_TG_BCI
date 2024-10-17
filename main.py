import pickle
import customtkinter as ctk
import numpy as np
import acquisition
import classifier

sampling_rate = 250
class InitialSelection(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BCI Initial Selection")
        self.geometry("400x300")

        self.label = ctk.CTkLabel(self, text="Selecione o Modo:")
        self.label.pack(pady=20)

        self.training_button = ctk.CTkButton(self, text="Treinamento", command=self.start_training)
        self.training_button.pack(pady=10)

        self.online_button = ctk.CTkButton(self, text="Aquisição Online", command=self.start_online)
        self.online_button.pack(pady=10)


    def start_training(self):
        self.destroy()
        app = TrainingApp()
        app.mainloop()

    def start_online(self):
        self.destroy()
        app = OnlineApp()
        app.mainloop()

class TrainingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BCI Training")
        self.geometry("800x600")

        self.label_training = ctk.CTkLabel(self, text="Rotina de Treinamento:")
        self.label_training.pack(pady=10)

        self.freq_entry = ctk.CTkEntry(self, placeholder_text="Frequências Desejadas (Hz)")
        self.freq_entry.pack(pady=10)

        self.trainningTime_entry = ctk.CTkEntry(self, placeholder_text="Tempo de Treinamento (s)")
        self.trainningTime_entry.pack(pady=10)

        self.channels_entry = ctk.CTkEntry(self, placeholder_text="Canais")
        self.channels_entry.pack(pady=10)

        self.trials_entry = ctk.CTkEntry(self, placeholder_text="Trials")
        self.trials_entry.pack(pady=10)

        self.confirm_button = ctk.CTkButton(self, text="Confirmar", command=self.confirm_selection)
        self.confirm_button.pack(pady=20)


    def confirm_selection(self):
        # frequencies = list(map(int, self.freq_entry.get().split(',')))
        # trainningTime = int(self.trainningTime_entry.get())
        # channels = int(self.channels_entry.get())
        # trials = int(self.trials_entry.get())
        frequencies = [8, 10, 12, 15]
        trainningTime = 5
        channels = 8
        trials = 6
        data_eeg = np.zeros((8, trainningTime*trials*sampling_rate,len(frequencies)), dtype=object)
        for freq in frequencies:
            for trial in range(trials):
                data = acquisition.trainningAcquisition(trainningTime, trial)
                data_eeg[:, trial*sampling_rate*trainningTime:(trial+1)*sampling_rate*trainningTime, frequencies.index(freq)] = data

                messagebox = ctk.CTkLabel(self, text=f"{freq} Hz - Trial {trial + 1} de {trials} concluído. Pressione Confirmar para iniciar o próximo trial.")
                messagebox.pack(pady=10)
                confirm_next_trial_button = ctk.CTkButton(self, text="Confirmar", command=messagebox.destroy)
                confirm_next_trial_button.pack(pady=10)
                confirm_next_trial_button._clicked = ctk.IntVar()
                confirm_next_trial_button.configure(command=lambda: confirm_next_trial_button._clicked.set(1))
                self.wait_variable(confirm_next_trial_button._clicked)
                confirm_next_trial_button.destroy()
                messagebox.destroy()

        #Problem with the data_eeg shape
        w, acc = classifier.buildWForOnline(data_eeg, channels, frequencies, sampling_rate, trainningTime, trials)
        acc_label = ctk.CTkLabel(self, text="Acurácia do Classificador por Frequência:")
        acc_label.pack(pady=10)
        for i, freq in enumerate(frequencies):
            freq_acc_label = ctk.CTkLabel(self, text=f"Frequência {freq} Hz: {acc[i]:.2f}%")
            freq_acc_label.pack(pady=5)
        

        # Salvar o classificador 'w' para uso posterior
        with open("classifier.pkl", "wb") as f:
            pickle.dump(w, f)
        
        back_button = ctk.CTkButton(self, text="Voltar", command=self.destroy)
        back_button.pack(pady=20)
        back_button._clicked = ctk.IntVar()
        back_button.configure(command=lambda: back_button._clicked.set(1))
        self.wait_variable(back_button._clicked)

        self.destroy()
        app = InitialSelection()
        app.mainloop()

class OnlineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BCI Online Acquisition")
        self.geometry("800x600")

        self.label_online = ctk.CTkLabel(self, text="Aquisição Online:")
        self.label_online.pack(pady=10)

        self.trainningTime_entry = ctk.CTkEntry(self, placeholder_text="Tempo de Aquisição (s)")
        self.trainningTime_entry.pack(pady=10)

        self.freq_entry = ctk.CTkEntry(self, placeholder_text="Frequências Desejadas (Hz)")
        self.freq_entry.pack(pady=10)

        self.channels_entry = ctk.CTkEntry(self, placeholder_text="Canais")
        self.channels_entry.pack(pady=10)

        self.confirm_button = ctk.CTkButton(self, text="Confirmar", command=self.confirm_selection)
        self.confirm_button.pack(pady=20)

    def confirm_selection(self):
        # trainningTime = int(self.trainningTime_entry.get())
        trainningTime = 5
        trials = 1
        frequencies = [8]
        channels = 8

        # Carregar o classificador salvo
        with open("classifier.pkl", "rb") as f:
            w = pickle.load(f)
        data_eeg = acquisition.BCIOnline(trainningTime, w)
        y_pred = classifier.classify(w, data_eeg, channels, frequencies, sampling_rate, trainningTime, trials)
        print(y_pred)

if __name__ == "__main__":
    app = InitialSelection()
    app.mainloop()