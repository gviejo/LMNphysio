# -*- coding: utf-8 -*-
# Shared writer for the Source Data file: one sheet per figure in a single Excel file.
import os
import pandas as pd
import openpyxl

SOURCE_DATA = "/home/gviejo/Dropbox/LMNphysio/paperfinal/Source_Data.xlsx"


def write_figure_sheet(sheet, blocks, descriptions, outfile=SOURCE_DATA):
    """Write all blocks of one figure to `sheet`, replacing it if it exists.

    Blocks are placed side by side, separated by an empty column.
    Row 1 = block name, row 2 = description, row 4 = column names, data from row 5.
    """
    if os.path.exists(outfile):
        # Remove the old sheet, then write all blocks onto the same new sheet
        wb = openpyxl.load_workbook(outfile)
        if sheet in wb.sheetnames:
            del wb[sheet]
            wb.save(outfile)
        writer = pd.ExcelWriter(outfile, engine="openpyxl", mode="a", if_sheet_exists="overlay")
    else:
        writer = pd.ExcelWriter(outfile, engine="openpyxl")
    with writer:
        col = 0
        for name, df in sorted(blocks.items()):
            df.to_excel(writer, sheet_name=sheet, startrow=3, startcol=col, index=False)
            ws = writer.sheets[sheet]
            ws.cell(row=1, column=col + 1, value=name.replace("_", " "))
            ws.cell(row=2, column=col + 1, value=descriptions[name])
            col += df.shape[1] + 1

    # Keep sheets in figure order (Fig1, Fig2, ...)
    wb = openpyxl.load_workbook(outfile)
    wb._sheets.sort(key=lambda ws: (len(ws.title), ws.title))
    wb.save(outfile)
    print("Saved", sheet, "in", outfile)
