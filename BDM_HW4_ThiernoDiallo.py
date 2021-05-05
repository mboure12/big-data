import functools
import datetime
import csv
import json
from pyspark.sql.functions import *
import pyspark

sc = pyspark.SparkContext()
spark = pyspark.sql.SparkSession.builder.appName('rddtodf').getOrCreate()

NYC_CITIES = set(['New York', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
food_store = {'big_box_grocers': ['452210', '452311'],
              'convenience_stores': ['445120'],
              'drinking_places': ['722410'],
              'full_service_restaurants': ['722511'],
              'limited_service_restaurants': ['722513'],
              'pharmacies_and_drug_stores': ['446110', '446191'],
              'snack_and_bakeries': ['311811', '722515'],
              'specialty_food_stores': ['445210', '445220', '445230', '445291', '445292', '445299'],
              'supermarkets_except_convenience_stores': ['445110']}


def listDates(stores, _, lines):
    toDateTime = lambda x: datetime.datetime(*map(int, x[:10].split('-')))
    # fromDate = toDateTime(fromStr)
    for row in csv.reader(lines):
        if row[12] != 'date_range_start' and row[1] in stores:
            startDate = toDateTime(row[12])
            year = startDate.year
            visits_by_day = json.loads(row[16])
            for i in range(len(visits_by_day)):
                yield (str(year) + ',' + (startDate + datetime.timedelta(i)).strftime("%Y-%m-%d")
                       , visits_by_day[i])


def get_year(date_str):
    return date_str.split(',')[0]


def get_date(date_str):
    return date_str.split(',')[1]


def countClosedRest(store_type, cusips, NYC_CITIES):
    stores = set(sc.textFile(sys.argv[1] if len(sys.argv)>1 else 'data/share/bdm/core-places-nyc.csv') \
                 .map(lambda x: x.split(',')) \
                 .map(lambda x: (x[1], x[9], x[13])) \
                 .filter(lambda x: (x[1] in cusips) and (x[2] in NYC_CITIES)) \
                 .map(lambda x: x[0]) \
                 .collect())

    udfgetYear = udf(get_year, StringType())
    udfgetDate = udf(get_date, StringType())

    rdd = sc.textFile(sys.argv[1] if len(sys.argv)>1 else 'data/share/bdm/weekly-patterns-nyc-2019-2020/*') \
        .mapPartitionsWithIndex(functools.partial(listDates, stores))

    if not rdd.isEmpty():
        spark.createDataFrame(rdd, ['datestr', 'daily_visits']) \
            .withColumn('Year', udfgetYear('datestr')) \
            .withColumn('Date', udfgetDate('datestr')) \
            .select('Year', 'Date', 'daily_visits') \
            .groupby('Date').agg(expr('percentile(daily_visits, array(0.00))')[0].alias('Min'),
                                 expr('percentile(daily_visits, array(0.50))')[0].alias('Median'),
                                 expr('percentile(daily_visits, array(1.00))')[0].alias('Max')) \
            .orderBy('Date') \
            .write.csv(sys.argv[2] if len(sys.argv)>2 else store_type + '.csv')


for store_type in food_store:
    countClosedRest(store_type, food_store[store_type], NYC_CITIES)