CSV_FIELDS = ['file_name', 'text', 'heat_id', 'heat_year', 'sheet_id']


def csv_record(result, file_name=None):
    return dict(file_name=file_name or result.get('file_name') or result.get('filename', ''),
                text=result.get('text', ''), heat_id=result.get('heat_id', ''),
                heat_year=result.get('heat_year') or '', sheet_id=result.get('sheet_id', ''))
