# Function to create pie chart
# WORKING but very difficult to style


def create_pie_chart(labels, sizes, colors, chart_size=(1, 1), dpi=600, label_font_size=4, auto_pct_size=4):
    """
    Create a pie chart with adjustable size and resolution.

    Parameters:
        labels (list): List of labels for the pie chart.
        sizes (list): Sizes (percentages) for the pie chart.
        colors (list): Colors for the pie chart sections.
        chart_size (tuple): Tuple of (width, height) to control chart size.
        dpi (int): Resolution of the chart (higher = better quality).
        label_font_size (int): Font size for labels on the chart.
        auto_pct_size (int): Font size for the percentage text inside the chart.

    Returns:
        matplotlib.figure.Figure: The generated pie chart figure.
    """
    fig, ax = plt.subplots(figsize=chart_size, dpi=dpi)  # Control size and resolution
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        textprops=dict(color="black", fontsize=label_font_size),  # Label size and color
    )
    plt.setp(autotexts, size=auto_pct_size, weight="bold")  # Smaller percentage text inside the pie
    for wedge in wedges:
        wedge.set_edgecolor('white')  # Clean edges for the pie chart
    return fig
