import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def plotLossPerEpoch(epochs:np.ndarray[int], training_loss:np.ndarray[float], testing_loss:np.ndarray[float], ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    
    ax.plot(np.array(epochs),np.array(training_loss), '+-b', label='Training loss', markersize=10)
    ax.plot(np.array(epochs),np.array(testing_loss), '+-r', label='Testing loss', markersize=10)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")
    ax.legend()

    return fig
