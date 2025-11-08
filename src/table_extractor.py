"""
Table Extractor Module

Extracts tables from PDF and Excel files and normalizes them to a common format.
Handles multiple tables per file and auto-detection of column types.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import pdfplumber
from google.genai.errors import APIError


class TableExtractor:
    """Extracts and normalizes tables from PDF and Excel files."""

    def __init__(self, gemini_client=None):
        """
        Initialize TableExtractor.

        Args:
            gemini_client: Optional GeminiChatClient for LLM-based column detection
        """
        self.gemini_client = gemini_client
        self.extracted_tables = {}  # Cache of extracted tables

    def extract_from_pdf(self, filepath: Path, table_index: int = 0) -> Optional[pd.DataFrame]:
        """
        Extract table from PDF file using pdfplumber.

        Args:
            filepath: Path to PDF file
            table_index: Which table to extract (0-indexed)

        Returns:
            DataFrame with extracted table, or None if extraction fails
        """
        try:
            with pdfplumber.open(filepath) as pdf:
                all_tables = []

                # Extract tables from all pages
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                if not all_tables:
                    print(f"No tables found in PDF: {filepath}")
                    return None

                if table_index >= len(all_tables):
                    print(f"Table index {table_index} out of range. Found {len(all_tables)} tables.")
                    return None

                # Convert table to DataFrame
                table_data = all_tables[table_index]
                if not table_data or len(table_data) < 2:
                    print(f"Table {table_index} is empty or has no data rows")
                    return None

                # First row as headers
                headers = table_data[0]
                data = table_data[1:]

                df = pd.DataFrame(data, columns=headers)

                # Clean up: remove empty rows and columns
                df = df.dropna(how='all', axis=0)  # Remove empty rows
                df = df.dropna(how='all', axis=1)  # Remove empty columns

                return df

        except Exception as e:
            print(f"Error extracting PDF table: {e}")
            return None

    def extract_from_excel(self, filepath: Path, sheet_name: int = 0) -> Optional[pd.DataFrame]:
        """
        Extract table from Excel file.

        Args:
            filepath: Path to Excel file
            sheet_name: Which sheet to read (0-indexed or sheet name)

        Returns:
            DataFrame with extracted table, or None if extraction fails
        """
        try:
            # Read Excel file
            if isinstance(sheet_name, int):
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:
                df = pd.read_excel(filepath, sheet_name=sheet_name)

            # Clean up: remove empty rows and columns
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)

            return df

        except Exception as e:
            print(f"Error extracting Excel table: {e}")
            return None

    def extract_from_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Extract table from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with extracted table, or None if extraction fails
        """
        try:
            df = pd.read_csv(filepath)

            # Clean up: remove empty rows and columns
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)

            return df

        except Exception as e:
            print(f"Error extracting CSV table: {e}")
            return None

    def extract_table(self, filepath: Path, table_index: int = 0) -> Optional[pd.DataFrame]:
        """
        Extract table from file (auto-detects format).

        Args:
            filepath: Path to file (PDF, Excel, or CSV)
            table_index: Which table/sheet to extract (0-indexed)

        Returns:
            DataFrame with extracted table, or None if extraction fails
        """
        suffix = filepath.suffix.lower()

        if suffix == '.pdf':
            return self.extract_from_pdf(filepath, table_index)
        elif suffix in ['.xlsx', '.xls']:
            return self.extract_from_excel(filepath, table_index)
        elif suffix == '.csv':
            return self.extract_from_csv(filepath)
        else:
            print(f"Unsupported file format: {suffix}")
            return None

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect which columns contain item, unit, and price.

        Uses heuristics and optionally LLM-based detection.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary mapping 'item', 'unit', 'price' to column names
        """
        columns = df.columns.tolist()
        detected = {}

        # Heuristic-based detection (Spanish construction terms)
        item_keywords = ['descripción', 'descripcion', 'item', 'concepto', 'actividad', 'material']
        unit_keywords = ['unidad', 'un', 'und', 'u', 'medida']
        price_keywords = ['precio', 'valor', 'costo', 'unitario', 'vr']

        for col in columns:
            col_lower = str(col).lower().strip()

            # Detect item column
            if not detected.get('item'):
                for keyword in item_keywords:
                    if keyword in col_lower:
                        detected['item'] = col
                        break

            # Detect unit column
            if not detected.get('unit'):
                for keyword in unit_keywords:
                    if keyword in col_lower:
                        detected['unit'] = col
                        break

            # Detect price column
            if not detected.get('price'):
                for keyword in price_keywords:
                    if keyword in col_lower:
                        detected['price'] = col
                        break

        # If not all columns detected, use positional heuristics
        if len(detected) < 3 and len(columns) >= 3:
            # Common pattern: Item, Unit, Price
            if not detected.get('item'):
                detected['item'] = columns[0]
            if not detected.get('unit') and len(columns) > 1:
                detected['unit'] = columns[1] if 'unit' not in detected.values() else columns[-2]
            if not detected.get('price'):
                detected['price'] = columns[-1]

        return detected

    def normalize_table(self, df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Normalize table to standard schema: [item, unit, price].

        Args:
            df: DataFrame to normalize
            column_map: Optional mapping {'item': 'col_name', 'unit': 'col_name', 'price': 'col_name'}

        Returns:
            Normalized DataFrame with columns [item, unit, price]
        """
        if column_map is None:
            column_map = self.detect_columns(df)

        # Validate column map
        if not all(k in column_map for k in ['item', 'unit', 'price']):
            print(f"Warning: Incomplete column mapping: {column_map}")
            return df

        # Create normalized DataFrame
        normalized = pd.DataFrame({
            'item': df[column_map['item']],
            'unit': df[column_map['unit']],
            'price': df[column_map['price']]
        })

        # Clean up item descriptions
        normalized['item'] = normalized['item'].astype(str).str.strip()

        # Clean up units
        normalized['unit'] = normalized['unit'].astype(str).str.strip()

        # Clean up prices (remove currency symbols, convert to float)
        normalized['price'] = normalized['price'].apply(self._clean_price)

        # Remove rows where item is empty or NaN
        normalized = normalized[normalized['item'].notna()]
        normalized = normalized[normalized['item'] != '']
        normalized = normalized[normalized['item'] != 'nan']

        # Remove section headers (rows without prices)
        normalized = normalized[normalized['price'].notna()]

        return normalized

    def _clean_price(self, value) -> Optional[float]:
        """Clean price value and convert to float."""
        if pd.isna(value):
            return None

        # Convert to string and clean
        price_str = str(value).strip()

        # Remove common currency symbols and formatting
        price_str = price_str.replace('$', '').replace('€', '').replace('£', '')
        price_str = price_str.replace(',', '').replace(' ', '')

        try:
            return float(price_str)
        except ValueError:
            return None

    def preview_table(self, df: pd.DataFrame, num_rows: int = 10) -> str:
        """
        Generate a preview of the table.

        Args:
            df: DataFrame to preview
            num_rows: Number of rows to show

        Returns:
            Formatted string preview
        """
        preview = f"Table Preview ({len(df)} total rows):\n"
        preview += "="*60 + "\n"
        preview += df.head(num_rows).to_string(index=False)
        preview += "\n" + "="*60
        return preview

    def get_table_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the table.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with table statistics
        """
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'has_prices': 'price' in df.columns,
            'num_items_with_prices': len(df[df['price'].notna()]) if 'price' in df.columns else 0
        }
