from io import BytesIO
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import io

pdf = FPDF()
pdf.add_page()

DATA = {
    "Unemployment_Rate": [6.1, 5.8, 5.7, 5.7, 5.8, 5.6, 5.5, 5.3, 5.2, 5.2],
    "Stock_Index_Price": [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565],
}
COLUMNS = tuple(DATA.keys())

plt.figure()  # Create a new figure object
df = pd.DataFrame(DATA, columns=COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind="scatter")

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi=200)  # Save the image

DATA = {
    "Unemployment_Rate": [6.1, 5.8, 5.7, 5.7, 5.8, 5.6, 5.5, 5.3, 5.2, 5.2],
    "Stock_Index_Price": [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565],
}
COLUMNS = tuple(DATA.keys())

plt.figure()  # Create a new figure object
df = pd.DataFrame(DATA, columns=COLUMNS)
df.plot(x = COLUMNS[0], y = COLUMNS[1], kind="scatter")

# Converting Figure to an image:
img_buf = BytesIO()  # Create image object
plt.savefig(img_buf, dpi = 200)  # Save the image

pdf.output("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion/Grafico.pdf")
img_buf.close()