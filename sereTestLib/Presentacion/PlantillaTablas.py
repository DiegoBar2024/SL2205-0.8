import pandas as pd
from fpdf import FPDF

class PDFTable:
    def __init__(self, title: str, data: pd.DataFrame):
        """
        Initialize the PDF with a title and the DataFrame to convert into a table.
        :param title: Title for the PDF document.
        :param data: Pandas DataFrame that holds the data to be displayed in the table.
        """
        self.title = title
        self.data = data
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font("Times", size=16)

    def add_title(self):
        """Adds the title to the PDF."""
        self.pdf.set_font("Times", style="B", size=18)
        self.pdf.cell(0, 10, self.title, ln=True, align="C")
        self.pdf.ln(10)  # Add a line break

    def add_table(self):
        """Adds the table from the DataFrame to the PDF."""
        # Create the table
        with self.pdf.table() as table:
            # Add headers (column names)
            header = self.data.columns.tolist()
            row = table.row()
            for column in header:
                row.cell(column)

            # Add rows (data)
            for index, data_row in self.data.iterrows():
                row = table.row()
                for item in data_row:
                    row.cell(str(item))

    def save_pdf(self, filename: str):
        """Generates the PDF and saves it to a file."""
        self.pdf.output(filename)

# Example usage

# Sample data for the report (Pandas DataFrame)
data = {
    "First name": ["Jules", "Mary", "Carlson", "Lucas"],
    "Last name": ["Smith", "Ramos", "Banks", "Cimon"],
    "Age": [34, 45, 19, 31],
    "City": ["San Juan", "Orlando", "Los Angeles", "Saint-Mathurin-sur-Loire"]
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Create an instance of the PDFTable class
pdf_table = PDFTable(title="Employee Report", data=df)

# Add title and table to the PDF
pdf_table.add_title()
pdf_table.add_table()

# Save the PDF to a file
pdf_table.save_pdf("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Presentacion/Tabla.pdf")