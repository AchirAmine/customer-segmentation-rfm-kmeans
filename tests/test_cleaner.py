import pandas as pd
import pytest
from src.data.cleaner import clean_retail_data


def test_clean_retail_data_cancellations():
    raw_data = pd.DataFrame(
        {
            "Invoice": ["489434", "C489435", "489436"],
            "StockCode": ["85048", "79323P", "79323W"],
            "Description": ["Item A", "Item B", "Item C"],
            "Quantity": [10, -5, 2],
            "InvoiceDate": ["2010-01-01", "2010-01-02", "2010-01-03"],
            "Price": [2.5, 3.0, 1.5],
            "Customer ID": [13085.0, 13085.0, 13086.0],
            "Country": ["United Kingdom", "United Kingdom", "France"],
        }
    )
    result = clean_retail_data(raw_data)
    assert len(result) == 2
    assert "C489435" not in result["Invoice"].values


def test_clean_retail_data_missing_customer():
    raw_data = pd.DataFrame(
        {
            "Invoice": ["489434", "489435"],
            "StockCode": ["85048", "79323P"],
            "Description": ["Item A", "Item B"],
            "Quantity": [10, 5],
            "InvoiceDate": ["2010-01-01", "2010-01-02"],
            "Price": [2.5, 3.0],
            "Customer ID": [13085.0, None],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )
    result = clean_retail_data(raw_data)
    assert len(result) == 1
    assert result["Customer ID"].iloc[0] == 13085


def test_clean_retail_data_non_positive_values():
    raw_data = pd.DataFrame(
        {
            "Invoice": ["489434", "489435", "489436"],
            "StockCode": ["85048", "79323P", "79323W"],
            "Description": ["Item A", "Item B", "Item C"],
            "Quantity": [10, 0, 5],
            "InvoiceDate": ["2010-01-01", "2010-01-02", "2010-01-03"],
            "Price": [2.5, 1.0, 0.0],
            "Customer ID": [13085.0, 13086.0, 13087.0],
            "Country": ["United Kingdom", "United Kingdom", "France"],
        }
    )
    result = clean_retail_data(raw_data)
    assert len(result) == 1
    assert result["TotalAmount"].iloc[0] == 25.0


def test_clean_retail_data_missing_column():
    raw_data = pd.DataFrame({"Invoice": ["489434"]})
    with pytest.raises(ValueError):
        clean_retail_data(raw_data)
