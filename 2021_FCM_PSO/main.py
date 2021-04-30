import particleSwarmOptimization
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.set_loglevel("info")

    cost, lam, anomaly_score = particleSwarmOptimization.pso()

    plt.subplot(212)
    plt.plot(anomaly_score)
    plt.show()

