import matplotlib.pyplot as plt


matplotlib.use("TkAgg")
plt.ion()


class LivePlotter:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.lines = {
            "policy_loss": self.axs[0, 0].plot([], [])[0],
            "value_loss": self.axs[0, 1].plot([], [])[0],
            "entropy": self.axs[1, 0].plot([], [])[0],
            "avg_return": self.axs[1, 1].plot([], [])[0],
        }
        self.axs[0, 0].set_title("Policy Loss")
        self.axs[0, 1].set_title("Value Loss")
        self.axs[1, 0].set_title("Entropy")
        self.axs[1, 1].set_title("Average Return")
        for ax in self.axs.flat:
            ax.set_xlabel("Iteration")
        self.fig.tight_layout()

    def update(self, history):
        for key, line in self.lines.items():
            y = history[key]
            x = list(range(1, len(y) + 1))
            line.set_data(x, y)
            ax = line.axes
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
