import re

with open('performance_report.py', 'r') as f:
    content = f.read()

# Add the safe wrapper function right after pdf init
insert_point = content.find('pdf.set_font("Helvetica", size=12)')
if insert_point != -1:
    end_of_line = content.find('\n', insert_point)
    wrapper_code = '''
        
        # Wrapper to ensure X position is reset before multi_cell
        def write_text(text, height=5):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, height, text)
'''
    content = content[:end_of_line+1] + wrapper_code + content[end_of_line+1:]

# Replace all pdf.multi_cell calls with write_text
content = re.sub(r'pdf\.multi_cell\(0,\s*5,', 'write_text(', content)

with open('performance_report.py', 'w') as f:
    f.write(content)

print("Fixed!")
