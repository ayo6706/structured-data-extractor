from datetime import date
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

MONEY_QUANT = Decimal("0.01")


def normalize_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANT)


def validate_currency_code(value: str | None) -> str | None:
    if value is None:
        return value

    normalized = value.upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError("currency must be a three-letter ISO 4217 code")

    return normalized


class LineItem(BaseModel):
    description: str = Field(description="Line item description")
    quantity: Decimal | None = Field(None, description="Quantity if stated")
    unit_price: Decimal | None = Field(None, description="Price per unit")
    amount: Decimal = Field(description="Total for this line item")


class Party(BaseModel):
    name: str = Field(description="Full legal name of the party")
    role: str | None = Field(
        None, description="e.g. 'Employer', 'Contractor', 'Landlord'"
    )
    address: str | None = Field(None, description="Full address if stated")


class Deduction(BaseModel):
    name: str = Field(
        description="Deduction name e.g. 'Tax', 'Pension', 'Health Insurance'"
    )
    amount: Decimal = Field(description="Deduction amount")


class InvoiceSchema(BaseModel):
    vendor_name: str = Field(description="Name of the vendor or supplier")
    vendor_address: str | None = Field(
        None, description="Address of the vendor or supplier"
    )
    invoice_number: str = Field(description="Unique invoice number")
    invoice_date: date = Field(description="Date the invoice was issued")
    due_date: date | None = Field(None, description="Date the payment is due")
    subtotal: Decimal = Field(description="Total before taxes and discounts")
    tax_amount: Decimal | None = Field(None, description="Total tax amount")
    total_amount: Decimal = Field(description="Final total amount due")
    currency: str = Field(
        description="Three-letter currency code, e.g., USD, EUR"
    )
    line_items: list[LineItem] = Field(
        description="List of items on the invoice"
    )
    payment_terms: str | None = Field(
        None, description="Payment terms if stated"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return validate_currency_code(value)

    @model_validator(mode="after")
    def validate_dates_and_totals(self) -> Self:
        if self.due_date is not None and self.due_date < self.invoice_date:
            raise ValueError("due_date must be on or after invoice_date")

        if self.tax_amount is not None:
            expected_total = normalize_money(self.subtotal + self.tax_amount)
            if normalize_money(self.total_amount) != expected_total:
                raise ValueError(
                    "total_amount must equal subtotal plus tax_amount"
                )

        return self


class ContractSchema(BaseModel):
    parties: list[Party] = Field(description="Parties involved in the contract")
    effective_date: date = Field(
        description="Date the contract becomes effective"
    )
    termination_date: date | None = Field(
        None, description="Date the contract expires or terminates"
    )
    governing_law: str | None = Field(
        None, description="Jurisdiction governing the contract"
    )
    payment_amount: Decimal | None = Field(
        None, description="Total payment or consideration amount"
    )
    payment_currency: str | None = Field(
        None, description="Three-letter currency code for the payment"
    )
    notice_period_days: int | None = Field(
        None, description="Required notice period in days for termination"
    )
    key_obligations: list[str] = Field(
        description="List of key obligations or deliverables"
    )

    @field_validator("payment_currency")
    @classmethod
    def validate_payment_currency(cls, value: str | None) -> str | None:
        return validate_currency_code(value)

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if (
            self.termination_date is not None
            and self.termination_date < self.effective_date
        ):
            raise ValueError(
                "termination_date must be on or after effective_date"
            )

        return self


class PayslipSchema(BaseModel):
    employee_name: str = Field(description="Name of the employee")
    employer_name: str = Field(description="Name of the employer")
    pay_period_start: date = Field(description="Start date of the pay period")
    pay_period_end: date = Field(description="End date of the pay period")
    gross_pay: Decimal = Field(description="Gross pay amount before deductions")
    net_pay: Decimal = Field(
        description="Final net pay amount after deductions"
    )
    currency: str = Field(description="Three-letter currency code")
    deductions: list[Deduction] = Field(description="List of deductions")
    pay_date: date = Field(description="Date the payment is made")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return validate_currency_code(value)

    @model_validator(mode="after")
    def validate_period_and_pay(self) -> Self:
        if self.pay_period_end < self.pay_period_start:
            raise ValueError(
                "pay_period_end must be on or after pay_period_start"
            )

        total_deductions = sum(
            (deduction.amount for deduction in self.deductions),
            start=Decimal("0"),
        )
        expected_net_pay = normalize_money(self.gross_pay - total_deductions)
        if normalize_money(self.net_pay) != expected_net_pay:
            raise ValueError(
                "net_pay must equal gross_pay minus total deductions"
            )

        return self


class ReceiptSchema(BaseModel):
    merchant_name: str = Field(description="Name of the merchant")
    merchant_address: str | None = Field(
        None, description="Address of the merchant"
    )
    receipt_date: date = Field(description="Date the receipt was issued")
    items: list[LineItem] = Field(description="List of items purchased")
    subtotal: Decimal | None = Field(None, description="Total before taxes")
    tax_amount: Decimal | None = Field(None, description="Total tax amount")
    total_amount: Decimal = Field(description="Final total amount paid")
    currency: str = Field(description="Three-letter currency code")
    payment_method: str | None = Field(
        None, description="Method of payment, e.g., 'Credit Card', 'Cash'"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return validate_currency_code(value)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.subtotal is None or self.tax_amount is None:
            return self

        expected_total = normalize_money(self.subtotal + self.tax_amount)
        if normalize_money(self.total_amount) != expected_total:
            raise ValueError("total_amount must equal subtotal plus tax_amount")

        return self
