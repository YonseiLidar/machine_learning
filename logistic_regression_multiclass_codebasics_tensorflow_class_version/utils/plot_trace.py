import matplotlib.pyplot as plt


def trace_plot(trace_data):

    beta0_trace, beta1_trace, beta2_trace, beta3_trace, loss_trace = trace_data

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
    ax0.plot(beta0_trace)
    ax0.set_xlabel('epoch')
    ax0.set_title('beta0')
    ax1.plot(beta1_trace)
    ax1.set_xlabel('epoch')
    ax1.set_title('beta1')
    ax2.plot(beta2_trace)
    ax2.set_xlabel('epoch')
    ax2.set_title('beta2')
    ax3.plot(beta3_trace)
    ax3.set_xlabel('epoch')
    ax3.set_title('beta3')
    ax4.plot(loss_trace)
    ax4.set_xlabel('epoch')
    ax4.set_title('loss')
    plt.tight_layout()
    plt.show()