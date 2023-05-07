import os
import sys
import threading
import time
import tkinter as tk
import tkinter.messagebox

import psutil
import win10toast


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


kill = False

battery = psutil.sensors_battery()
plugged = battery.power_plugged
percent = str(battery.percent)
plugged = "Plugged In" if plugged else "Not Plugged In"
print(percent + '% | ' + plugged)

global doLoop


# Main Gui
def shutdown():
    if psutil.sensors_battery().power_plugged:
        return
    threading.Thread(target=tkinter.messagebox.showwarning,
                     args=("AkkuGuard!", "Das System fährt in 3 Minuten herunter!\nSpeicher alles ab oder schließe "
                                         "deinen Computer an Strom an!")).start()
    timeWindow = tk.Tk()
    timeWindow.protocol("WM_DELETE_WINDOW", timeWindow.deiconify)
    timeWindow.attributes('-topmost', True)
    timeWindow.overrideredirect(True)
    timeWindow.update()
    timeWindow.eval('tk::PlaceWindow . center')
    timeWindow.attributes('-transparentcolor', 'white')
    timeLabel = tk.Label(timeWindow, bg="black", fg="red", font=("Arial", 20, "bold"))
    timeLabel.pack()
    timing = 60 * 2 * 3
    for i in range(timing):
        print(i)
        if i % 2 == 0:
            timeLabel["text"] = f"Verbleibende Zeit: {int((timing - i) / 2)}s"
            timeWindow.update()
        if psutil.sensors_battery().power_plugged:
            break
        if i == timing - 1 - 30 * 2:
            threading.Thread(target=tkinter.messagebox.showerror,
                             args=("AkkuGuard!", "Der Computer fährt in 30s herunter!")).start()
        if i >= timing - 1:
            print("Warning")
            break
        time.sleep(0.5)
    timeWindow.destroy()


class Main:
    def __init__(self):
        self.doLoop = None

    def setGui(self):
        self.doLoop = True
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.protocol("WM_DELETE_WINDOW", self.root.deiconify)
        self.root.configure(background='white')
        self.root.attributes('-transparentcolor', 'white')

        # Background Gui
        self.toplevel = tk.Toplevel(self.root)
        self.toplevel.attributes('-fullscreen', True)
        self.toplevel.lift(aboveThis=self.root)
        self.toplevel.protocol("WM_DELETE_WINDOW", self.toplevel.deiconify)
        self.toplevel.attributes('-alpha', 0.3)

        # Labels and Buttons
        self.dark_label15 = tk.Label(self.root, bg="black",
                                     anchor='n',
                                     font=("Arial", 20, "bold"))
        self.dark_label15.pack(side="top", expand=True, ipadx=self.root.winfo_screenwidth(),
                               ipady=self.root.winfo_screenheight() / 20)

        self.close_button = tk.Button(self.root, bg="light green", compound="center", width=50, height=3)
        self.close_button.place(relx=0.5, rely=0.52, anchor='center')

    def start15(self):
        self.setGui()
        self.dark_label15["text"] = "Schließe deinen Computer an!\nAkkuladung: 15%"
        self.dark_label15["fg"] = "grey"
        self.close_button["command"] = self.kill15
        self.close_button["text"] = "OK!"
        self.doLoop = True
        while True:
            if psutil.sensors_battery().power_plugged or not self.doLoop:
                break
            self.root.attributes('-topmost', True)
            self.root.update()
        try:
            self.root.destroy()
        except Exception as e:
            print(e)

    def kill15(self):
        self.root.destroy()
        self.doLoop = False

    def start20(self):
        toast = win10toast.ToastNotifier()
        toast.show_toast(
            "AkkuGuard!",
            "\nSchließe deinen Computer bitte an Strom an!\n\nAkkuladung: 20%",
            duration=8,
            icon_path="logo.ico",
            threaded=True
        )

    def kill10(self):
        self.doLoop = False
        self.root.destroy()
        shutdown()

    def start8(self):
        self.setGui()
        self.dark_label15["text"] = "Schließe deinen Computer an Strom an oder fahre ihn herunter!\nAkkuladung: 8%"
        self.dark_label15["fg"] = "red"
        self.close_button["command"] = self.kill10
        self.close_button["text"] = "Computer herunterfahren!"

        self.doLoop = True
        while True:
            if psutil.sensors_battery().power_plugged or not self.doLoop:
                if not self.doLoop:
                    return
                self.root.destroy()
                break
            self.root.attributes('-topmost', True)
            self.root.update()
        try:
            self.root.destroy()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print("DEBUG: program started")
    main = Main()
    while True:
        print("DEBUG: entered main-loop")
        if not psutil.sensors_battery().power_plugged:
            print(psutil.sensors_battery().percent)
            if 20 >= psutil.sensors_battery().percent >= 16:
                print("DEBUG: start 20")
                main.start20()
                while True:
                    print("DEBUG: wait for next battery change!")
                    if psutil.sensors_battery().power_plugged or not 20 >= psutil.sensors_battery().percent >= 15:
                        break
                    time.sleep(5)
            elif 15 >= psutil.sensors_battery().percent >= 8:
                print("DEBUG: start 15")
                main.start15()
                while True:
                    print("DEBUG: wait for next battery change!")
                    if psutil.sensors_battery().power_plugged or not 15 >= psutil.sensors_battery().percent >= 8:
                        break
                    time.sleep(5)
            elif psutil.sensors_battery().percent <= 8:
                print("DEBUG: start 8")
                main.start8()
                while True:
                    print("DEBUG: wait for next battery change!")
                    if psutil.sensors_battery().power_plugged or not psutil.sensors_battery().percent <= 8:
                        break
                    time.sleep(5)
        print("DEBUG: enter delay")
        time.sleep(30)
