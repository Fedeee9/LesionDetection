import os
import time
import config
import pandas as pd

if __name__ == '__main__':

    file = open('labels.txt', 'r')
    text = ''

    for f in file:
        x = f.strip()
        string = x.strip(",")
        text += string + "\n"

    newfile = open('finalLabelsTraining.txt', 'w')
    newfile.write(text)
    print("Stampa effettuata")
    newfile.close()
