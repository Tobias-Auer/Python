import os
import random
import subprocess
import time
import threading
import sys
import pygame
import pystray
from PIL import Image
from pystray import MenuItem as item, Menu as menu

FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')


class MusicPlayer:
    def __init__(self):
        self.kill_thread = False
        self.shuffle_mode = False
        pygame.mixer.init()
        self.paused = False
        self.playing = False

    def play_music(self, path, endless=True, action=None):
        # print("Play music")

        self.playing = True
        self.kill_thread = True
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        if endless:
            pygame.mixer.music.play(-1)
        else:
            self.kill_thread = False
            self.shuffle_mode = True
            pygame.mixer.music.play(0)
            # print("music")
            while pygame.mixer_music.get_busy() or self.paused:
                # print("DEBUG: " + str(pygame.mixer_music.get_busy()) + str(self.paused))
                # print("Waiting")
                time.sleep(0.5)
                if self.kill_thread:
                    # print("killing thread")
                    self.kill_thread = False
                    self.shuffle_mode = False
                    break
            if self.shuffle_mode:
                # print("next song................................")
                play_music(action)

    def pause_start_music(self):
        if self.playing:
            # print("action recognition")
            if self.paused:
                # print("start")
                pygame.mixer.music.unpause()
                self.paused = False
            else:
                # print("pause")
                pygame.mixer.music.pause()
                self.paused = True

    def stop_music(self):
        pygame.mixer.music.stop()
        self.kill_thread = True
        time.sleep(0.6)
        self.paused = False
        self.playing = False


# Funktion zum Beenden des Programms
def exit_program(icon, item):
    try:
        icon.stop()
        player.stop_music()
        sys.exit()
    except:
        ...


def get_file_path_from_index(index):
    dict = create_mp3_dict()
    # print("Dict: " + str(dict))
    entries = [item for sublist in dict.values() for item in sublist]

    # print("List: " + str(entries))
    # print("File: " + str(entries[int(index)]))
    return entries[int(index)]


def get_playlist_path_from_index(index):
    dict = create_mp3_dict()
    entries = list(dict.keys())
    # print("List: " + str(entries))
    # print("File: " + str(entries[int(index)]))
    return entries[int(index)]


def pause_start_music():
    global player
    player.pause_start_music()


def play_music(item):
    global player
    player.paused = False
    # print(item)
    index = str(item).split(":")[0]
    action = str(item).split(":")[1]
    if action == "Shuffle":
        player.kill_thread = True
        time.sleep(0.6)
        all_items = list()
        # print("shuffle list: " + index)
        # print(get_playlist_path_from_index(index))
        directory = get_playlist_path_from_index(index)
        directory = "./playlists/" + directory
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.mp3'):
                filepath = os.path.join(directory, file)
                all_items.append(filepath)
        # print("All items: " + str(all_items))
        choosed_item = random.choice(all_items)
        # print("Choosed Item: " + choosed_item)
        if player.shuffle_mode:
            player.play_music(choosed_item, False, item)
        else:
            threading.Thread(target=player.play_music, args=(choosed_item, False, item)).start()
    else:
        player.shuffle_activated = False
        # print(get_file_path_from_index(index))
        player.play_music(get_file_path_from_index(index))


def generate_menu_entries(mp3_files_dict):
    entries = []
    counter = 0
    list_counter = 0
    for playlist, mp3_files in mp3_files_dict.items():
        i = 0
        for _ in mp3_files:
            mp3_files[i] = str(counter) + ":" + mp3_files[i].split("\\")[-1].split(".mp3")[0]
            i += 1
            counter += 1
        playlist_items = [item(mp3_file, lambda icon, item: play_music(item)) for mp3_file in mp3_files]
        playlist_items.insert(0, item(f"{list_counter}:Shuffle", lambda icon, item: play_music(item)))
        playlist_menu = menu(*playlist_items)
        entries.append(item(playlist, playlist_menu))
        list_counter += 1
    return entries


def open_config_dir():
    path = os.path.normpath("./playlists")

    if os.path.isdir(path):
        subprocess.run([FILEBROWSER_PATH, path])
    elif os.path.isfile(path):
        subprocess.run([FILEBROWSER_PATH, '/select,', os.path.normpath(path)])


def create_mp3_dict():
    directory = './playlists'
    all_playlists = list([x[0] for x in os.walk(directory)])

    mp3_files_dict = {}
    for sub_folder in all_playlists:
        folder_name = os.path.basename(sub_folder)
        mp3_files = []
        files = os.listdir(sub_folder)
        for file in files:
            if file.endswith('.mp3'):
                filepath = os.path.join(sub_folder, file)
                mp3_files.append(filepath)
        if mp3_files:
            mp3_files_dict[folder_name] = mp3_files
    # print(mp3_files_dict)
    return mp3_files_dict


def main():
    # Erstelle das Systray-Icon
    image = Image.open("icon3.png")

    mp3_files_dict = create_mp3_dict()
    playing_label = item("Playing: ", lambda: None)

    menuEntries = menu(
        item("Pause/Start", pause_start_music, default=True),
        menu.SEPARATOR,
        *generate_menu_entries(mp3_files_dict),
        menu.SEPARATOR,
        item("Exit", exit_program),
        menu.SEPARATOR,
        item("Config", open_config_dir)
    )
    icon = pystray.Icon("MP3 Player", image, "MP3 Player", menu=menuEntries)
    try:
        icon.run()
    except KeyboardInterrupt:
        icon.stop()


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    player = MusicPlayer()
    main()
