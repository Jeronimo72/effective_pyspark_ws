import datetime

import holidays
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, dayofweek, udf
from pyspark.sql.types import BooleanType, DateType, StructField, StructType


def is_belgian_holiday(date: datetime.date) -> bool:
    belgian_holidays = holidays.BE()
    return date in belgian_holidays


def label_weekend(
    frame: DataFrame, colname: str = "date", new_colname: str = "is_weekend"
) -> DataFrame:
    # The line below is one solution. There is a more performant version of it
    # too, involving the modulo operator, but it is more complex. The
    # performance gain might not outweigh the cost of a programmer trying to
    # understand that arithmetic.

    # Keep in mind, that we're using "magic numbers" here too: unless you know
    # the API by heart, you're better off defining "SUNDAY = 1" and
    # "SATURDAY = 7" and using those identifiers in the call to the `isin`
    # method.
    return frame.withColumn(new_colname, dayofweek(col(colname)).isin(1, 7))


def label_holidays(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    """Add a column indicating whether or not the column `colname`
    is a holiday."""

    # holiday_udf = udf(lambda z: is_belgian_holiday(z), BooleanType())

    # If you were to see something like the line above in serious code, the
    # author of that line might not have understood the concepts of lambda
    # functions (nameless functions) and function references well. The
    # assignment above is more efficiently written as:
    holiday_udf = udf(is_belgian_holiday, BooleanType())

    return frame.withColumn(new_colname, holiday_udf(col(colname)))


def label_holidays2(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    """Add a column indicating whether or not the column `colname`
    is a holiday."""

    # A more efficient implementation of `label_holidays` than the udf-variant.
    # Major downside is that the range of years needs to be known a priori. Put
    # them in a config file or extract the range from the data beforehand.
    holidays_be = holidays.BE(years=list(range(2015, 2020)))
    return frame.withColumn(
        new_colname, col(colname).isin(list(holidays_be.keys()))
    )


def label_holidays3(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    """Add a column indicating whether or not the column `colname`
    is a holiday."""

    # Another more efficient implementation of `label_holidays`. Same downsides
    # as label_holidays2, but scales better.
    holidays_be = holidays.BE(years=list(range(2015, 2020)))
    spark = SparkSession.builder.getOrCreate()

    holidays_frame = spark.createDataFrame(
        data=[(day, True) for day in holidays_be.keys()],
        schema=StructType(
            [
                StructField(colname, DateType(), False),
                StructField(new_colname, BooleanType(), False),
            ]
        ),
    )

    return frame.join(holidays_frame, on=colname, how="left").na.fill(
        False, subset=[new_colname]
    )
