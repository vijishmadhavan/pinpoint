# Write & Create

## Can do
Create text files, Excel spreadsheets, charts (bar/line/pie/scatter/hist).

## Cannot do
Cannot create PDFs (use images_to_pdf), cannot create images from scratch (use run_python with PIL).

## Tools
- **write_file(path, content, append?)** → Create or write any text file. Set append=true to add to existing file. Use for: notes, reports, summaries, text exports, config files.
- **generate_excel(path, data, columns?, sheet_name?)** → Create Excel file from data. data is a list of rows (dicts or lists). Use for: expense reports, data exports, aggregated results.
- **generate_chart(data, chart_type, title?, xlabel?, ylabel?, output_path?)** → Create chart image. Types: bar, line, pie, scatter, hist. data format: {"labels": [...], "values": [...]}. Returns image path — send via send_file.

## Notes
- Paths must be absolute
- generate_excel creates .xlsx files using pandas + openpyxl
- generate_chart saves PNG at 150 DPI — good for WhatsApp
- Chain: analyze_data → extract numbers → generate_chart → send_file
