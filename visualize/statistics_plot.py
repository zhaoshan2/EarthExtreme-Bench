import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes
from matplotlib.ticker import ScalarFormatter


def plotSurfaceVariable(means, stds, name, ylabel, var):
    variables = ["Climatology", name]
    colors = ["#9edae5", "#ffbb78"]
    # X positions for the variables
    x = np.arange(len(variables))

    # Error bars (std deviations)
    errors = stds
    # Plotting
    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    # Plot mean values with error bars
    for i in range(len(variables)):
        ax.errorbar(
            x[i],
            means[i],
            yerr=errors[i],
            fmt="o",
            label=f"Mean ± Std ({variables[i]})",
            markersize=9,
            linewidth=4,
            color=colors[i],
            capsize=8,
        )
    ax.plot(x, means, linestyle="--", color="gray")
    # Adding labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    # Adding legend
    ax.legend(fontsize=13)  # loc='upper right'
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.tight_layout()
    # Display plot
    plt.savefig(f"figures/{var}_statistics.png", dpi=300)


def plotUpAirVariable(means, stds, name, ylabel, var):
    # X positions for the variables
    x = np.arange(len(means))

    # Error bars (std deviations)
    errors = stds
    colors = ["#9edae5", "#ffbb78"]
    # Plotting
    fig, bax = plt.subplots(figsize=(6.5, 6))

    # Plot mean values with error bars
    for i in range(len(means)):
        bax.errorbar(
            x[i],
            means[i],
            yerr=errors[i],
            fmt="o",
            markersize=9,
            linewidth=4,
            color=colors[i % 2],
            capsize=8,
        )
    # Connect mean values with dashed lines
    for i in range(0, len(means) - 1, 2):
        bax.plot(x[i : i + 2], means[i : i + 2], linestyle="--", color="gray")
    # Adding labels and title

    bax.set_xticks(x[::2])
    bax.set_xticklabels(
        ["200", "500", "700", "850", "1000"], fontsize=14, fontweight="bold"
    )

    bax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    bax.set_xlabel("Pressure levels / hPa", fontsize=18, fontweight="bold")
    # Adding legend
    bax.errorbar(
        [],
        [],
        yerr=errors[0],
        fmt="o",
        markersize=9,
        linewidth=4,
        color=colors[0],
        capsize=8,
        label="Mean ± Std (Climatology)",
    )
    bax.errorbar(
        [],
        [],
        yerr=errors[0],
        fmt="o",
        markersize=9,
        linewidth=4,
        color=colors[1],
        capsize=8,
        label=f"Mean ± Std ({name})",
    )
    bax.legend(fontsize=13)
    bax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    bax.spines["right"].set_visible(False)
    bax.spines["top"].set_visible(False)
    # bax.grid(axis='y')
    # Display plot
    plt.savefig(f"figures/{var}_statistics.png", dpi=300)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--varia", default="tc-z")
    args = parser.parse_args()

    if args.varia == "t2m-low":
        # coldwave  statistics
        means = [278.44345, 274.3225]
        stds = [21.501575, 13.1291]
        plotSurfaceVariable(
            means,
            stds,
            name="Coldwave",
            ylabel="Temperature 2 meters/ Kelvin",
            var=args.varia,
        )

    elif args.varia == "t2m-high":
        # heatwave  statistics
        means = [278.44345, 295.2374]
        stds = [21.501575, 9.2757]
        plotSurfaceVariable(
            means,
            stds,
            name="Heatwave",
            ylabel="Temperature 2 meters/ Kelvin",
            var=args.varia,
        )
    # cyclone
    elif args.varia == "tc-mslp":
        means = [100957.83, 101219.9688]
        stds = [1320.9279, 618.3453]
        plotSurfaceVariable(
            means,
            stds,
            name="Tropical cyclone",
            ylabel="Mean sea level pressure / Pa",
            var=args.varia,
        )

    elif args.varia == "tc-u10":
        means = [-0.050557569, -1.6686]  # [0.18899263, 0.4025]
        stds = [5.4929, 5.3051]  # [4.712085, 4.6408]
        plotSurfaceVariable(
            means,
            stds,
            name="Tropical cyclone",
            ylabel=r"10m U-component of wind / ms$^{-1}$",
            var=args.varia,
        )

    elif args.varia == "tc-v10":
        means = [0.18899263, 0.4025]
        stds = [4.712085, 4.6408]
        plotSurfaceVariable(
            means,
            stds,
            name="Tropical cyclone",
            ylabel=r"10 m V-component of wind / ms$^{-1}$",
            var=args.varia,
        )

    elif args.varia == "tc-u":
        means = [
            14.1896858,
            8.5617,
            6.55437517,
            2.9309,
            3.34081912,
            0.9672,
            1.41666412,
            -0.467,
            -0.033002265,
            -1.8037,
        ]
        stds = [
            17.6721344,
            16.9854,
            11.978776,
            10.2176,
            9.16401768,
            8.5022,
            8.18339729,
            7.8284,
            6.13872671,
            6.0425,
        ]
        plotUpAirVariable(
            means,
            stds,
            name="Tropical cyclone",
            ylabel=r"U-component of wind / m s$^{-1}$",
            var=args.varia,
        )

    elif args.varia == "tc-v":
        means = [
            -0.045016967,
            -0.9204,
            -0.023843139,
            0.6524,
            0.021518903,
            0.8711,
            0.142662525,
            0.8893,
            0.186560124,
            0.505,
        ]
        stds = [
            11.8723993,
            11.6322,
            9.17533016,
            6.8273,
            6.86629343,
            6.1917,
            6.26011753,
            6.0997,
            5.30454493,
            5.2954,
        ]
        plotUpAirVariable(
            means,
            stds,
            name="Tropical cyclone",
            ylabel=r"V-component of wind / m s$^{-1}$",
            var=args.varia,
        )

    elif args.varia == "tc-z":
        from brokenaxes import brokenaxes
        from matplotlib.ticker import ScalarFormatter

        means = [
            115558.266,
            121268.2969,
            54132.9375,
            57232.7578,
            28888.2793,
            30742.1094,
            13779.7188,
            14768.7148,
            737.141235,
            1038.7257,
        ]
        stds = [
            5832.26709,
            2051.2834,
            3357.07349,
            1083.3719,
            2138.3645,
            729.3351,
            1471.26855,
            560.0722,
            1072.25879,
            517.8634,
        ]

        # X positions for the variables
        x = np.arange(len(means))

        # Error bars (std deviations)
        errors = stds
        colors = ["#9edae5", "#ffbb78"]
        # Plotting
        fig = plt.figure(figsize=(8, 12))

        bax = brokenaxes(
            ylims=(
                (0, 2000),
                (10000, 16000),
                (25000, 35000),
                (50000, 60000),
                (110000, 125000),
            ),
            hspace=0.05,
        )
        # Set the y-axis formatter to scientific notation
        for ax_row in bax.axs:
            ax_row.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax_row.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # Plot mean values with error bars
        for i in range(len(means)):
            bax.errorbar(
                x[i],
                means[i],
                yerr=errors[i],
                fmt="o",
                markersize=9,
                linewidth=4,
                color=colors[i % 2],
                capsize=8,
            )
        # Connect mean values with dashed lines
        for i in range(0, len(means) - 1, 2):
            bax.plot(x[i : i + 2], means[i : i + 2], linestyle="--", color="gray")
        # Adding labels and title
        bax.errorbar(
            [],
            [],
            yerr=errors[0],
            fmt="o",
            markersize=9,
            linewidth=4,
            color=colors[0],
            capsize=8,
            label="Mean ± Std (Climatology)",
        )
        bax.errorbar(
            [],
            [],
            yerr=errors[0],
            fmt="o",
            markersize=9,
            linewidth=4,
            color=colors[1],
            capsize=8,
            label=f"Mean ± Std (Tropical cyclone)",
        )
        bax.legend(fontsize=16)
        bax.tick_params(axis="both", which="major", labelsize=14)
        # Display plot
        plt.savefig("figures/tc_geopotential_statistics.png", dpi=300)
