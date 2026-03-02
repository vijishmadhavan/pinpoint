# Data Analysis

## Can do
Analyze structured data in CSV/Excel: stats, groupby, filter, sort, custom pandas expressions, math.

## Cannot do
Cannot read PDFs, images, or unstructured text. For those use read_document, read_file, or ocr.

## Tools
- **analyze_data(path, operation, columns?, query?, head?)** → Run pandas operations on CSV or Excel files. Operations: describe (summary stats), head (first N rows), columns (list columns), value_counts (frequency), groupby (aggregate by column), filter (query rows), corr (correlation), sort (sort by column), unique (unique values), shape (dimensions), eval (custom pandas expression via query param). Returns: result as text/table.
- **read_excel(path, sheet_name?, cell_range?)** → Read specific cells/ranges from Excel. Direct cell access like "A1:D10", "B5", column "A:A". Returns: data as markdown table.
- **calculate(expression)** → Safe math evaluation. Supports: +, -, *, /, **, %, parentheses, round(), abs(), min(), max(), sum(), sqrt(), pi, e. Use for arithmetic on extracted numbers.

## Notes
- analyze_data for bulk operations (stats, groupby, filtering)
- eval operation: custom pandas expression via query param, e.g. `(df["Qty"] * df["Price"]).sum()`
- read_excel for specific cell access
- calculate for math on numbers you've already extracted
