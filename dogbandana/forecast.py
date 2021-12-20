import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast


class DogBandana:
    ''' Dog Bandana - Forecast weekly sales for BANDANA014 '''

    def __init__(self, product="BANDANA014"):
        self.product = product

    def read_orders(self, filename):
        ''' Read order data into pd dataframe '''
        orders = pd.read_csv(filename)
        orders.order_date = pd.to_datetime(orders.order_date)
        self.orders = orders

    def impute_missing_sales(self):
        ''' Impute orders when a product was not actively sold

        If a product was not live, use the other year of sales data to impute the missing data
        '''
        orders = self.orders.copy()
        b009_orders = orders[
            (orders.product_sku == "BANDANA009") &
            (orders.order_date > '2019-09-14') &
            (orders.order_date < '2019-12-31')
        ].copy()
        b014_orders = orders[
            (orders.product_sku == "BANDANA014") &
            (orders.order_date >= '2020-01-01') &
            (orders.order_date <= '2020-01-21')
        ].copy()
        b020_orders = orders[
            (orders.product_sku == "BANDANA020") &
            (orders.order_date >= '2020-01-01') &
            (orders.order_date <= '2020-11-28')
        ].copy()

        b009_orders.order_date = b009_orders.order_date.apply(
            lambda d: datetime.strptime('2020-' + datetime.strftime(d, '%Y-%m-%d')[5:], '%Y-%m-%d')
        )
        b014_orders.order_date = b014_orders.order_date.apply(
            lambda d: datetime.strptime('2019-' + datetime.strftime(d, '%Y-%m-%d')[5:], '%Y-%m-%d')
        )
        b020_orders.order_date = b020_orders.order_date.apply(
            lambda d: datetime.strptime('2019-' + datetime.strftime(d, '%Y-%m-%d')[5:], '%Y-%m-%d')
        )

        imputed_orders = pd.concat([orders, b009_orders, b014_orders, b020_orders])
        imputed_orders.sort_values(['product_sku', 'order_date'], inplace=True)

        self.imputed_orders = imputed_orders

    def shift_dates(self):
        '''
        Shift the sales history of Bandana009 and Bandana020 back in time

        so that we have one long ts from which to estimate seasonality
        '''

        orders = self.imputed_orders.copy()
        orders['year'] = orders.order_date.dt.isocalendar().year
        orders['week'] = orders.order_date.dt.isocalendar().week
        orders['day'] = orders.order_date.dt.isocalendar().day

        def _shift(row):
            if row['product_sku'] == 'BANDANA009':
                shift_year = row['year'] - 2
            elif row['product_sku'] == 'BANDANA020':
                shift_year = row['year'] - 4
            else:
                shift_year = row['year']
            return shift_year

        orders['shift_year'] = orders.apply(_shift, axis=1)
        orders['order_date'] = orders.apply(
            lambda row: f'{row["shift_year"]}-{row["week"]}-{row["day"]}',
            axis=1
        )
        orders.order_date = orders.order_date.apply(lambda d: datetime.strptime(d, "%G-%V-%u"))
        orders.order_date = pd.to_datetime(orders.order_date)

        orders.drop(columns=['year', 'week', 'day', 'shift_year'], inplace=True)
        orders.sort_values(['product_sku', 'order_date'], inplace=True)

        self.shifted_orders = orders

    def normalize_products(self):
        '''
        Normalize the product volumes
        '''

        orders = self.shifted_orders.copy()

        # product totals
        order_counts = orders.groupby(['product_sku']).agg(sku_total=('order_quantity', 'sum'))
        orders = orders.merge(order_counts, how='left', on='product_sku')
        orders.order_quantity = orders.order_quantity / orders.sku_total

        b14 = order_counts[order_counts.index == self.product]

        orders.order_quantity = orders.order_quantity * order_counts.sku_total[
            order_counts.index == self.product
        ][0]

        self.orders_norm = orders

    def smooth(self, time_grain='W'):
        '''
        Aggregate orders by time grain

        By default sum up the orders at a weekly grain
        '''

        orders = self.orders_norm.copy()

        beg_d = f'{orders.order_date.dt.year.min()}-01-01'
        end_d = f'{orders.order_date.dt.year.max()}-12-31'

        dr = pd.date_range(beg_d, end_d)

        index = pd.MultiIndex.from_product(
            [orders.product_sku.unique(), dr],
            names=['product_sku', 'order_date']
        )

        full_dr = pd.DataFrame(index=index).reset_index()
        orders = orders.merge(
            full_dr,
            how='outer',
            on=['product_sku', 'order_date']
        )
        orders.order_quantity = orders.order_quantity.fillna(0)
        orders.set_index('order_date', inplace=True)

        time_series = orders.order_quantity.resample(time_grain).sum()

        self.time_series = time_series

    def forecast(self, weeks=53):
        ''' Forecast for future weeks of the product

        Use STL for seasonality
        Use ARIMA for forecast of seasonally adjusted data

        '''
        ts = self.time_series
        stlf = STLForecast(
            ts,
            ARIMA,
            model_kwargs=dict(order=(1, 1, 0), trend="t")
        )
        stlf_res = stlf.fit()
        forecast = stlf_res.forecast(weeks)

        # Add various fields back into dataframe
        forecast.index.name = "order_date"
        forecast.name = "order_quantity"
        forecast = forecast.reset_index()
        forecast['id'] = np.NaN
        forecast['company'] = '4db8c0cdd9ef'
        forecast['product'] = 'b9da9eef0e7a'
        forecast['product_sku'] = 'BANDANA014'
        forecast['product_family_1'] = 'Toy'
        forecast['product_family_2'] = 'Toy'
        forecast['product_family_3'] = 'TOYS_AND_GAMES'
        forecast['forecast_type'] = 'forecast'
        forecast = forecast[[
            'id', 'company', 'product', 'product_sku', 'product_family_1',
            'product_family_2', 'product_family_3', 'order_quantity', 'order_date',
            'forecast_type'
        ]]

        self.forecast = forecast

    def combine_both(self):
        '''
        Combine Orders and forecast into a single weekly dataframe
        '''

        orders = self.orders.copy()
        orders['year'] = orders.order_date.dt.isocalendar().year
        orders['week'] = orders.order_date.dt.isocalendar().week
        week_orders = orders.groupby(
            ['product_sku', 'year', 'week']
        ).agg(order_quantity=('order_quantity', 'sum'))
        week_orders.reset_index(inplace=True)
        week_orders['forecast_type'] = 'sales'

        forecast = self.forecast.copy()
        forecast['year'] = forecast.order_date.dt.isocalendar().year
        forecast['week'] = forecast.order_date.dt.isocalendar().week

        keep_cols = ['product_sku', 'year', 'week', 'forecast_type', 'order_quantity']
        results = pd.concat(
            [week_orders[keep_cols], forecast[keep_cols]]
        ).sort_values(['product_sku', 'year', 'week'])

        return results

    def print_orders(self):
        print(self.forecast)

    def plot_forecast(self, results):
        '''
        Plot the Sales History and Forecast
        '''
        results['order_date'] = results.apply(
            lambda row: f'{row["year"]}-{row["week"]}-1',
            axis=1
        )
        results.order_date = pd.to_datetime(results.order_date, format='%G-%V-%u')

        unique_skus = results.product_sku.unique()
        colors = ['blue', 'red', 'green']
        fig, ax_list = plt.subplots(len(unique_skus), 1, figsize=(14, 10), sharex=True)

        # gridlines
        xcoords = ['2019-01-01', '2020-01-01', '2021-01-01']

        for i, sku in enumerate(unique_skus):
            sku_orders = results[results.product_sku == sku]
            sku_forecast = sku_orders[sku_orders['forecast_type'] == 'forecast']
            sku_orders = sku_orders[sku_orders['forecast_type'] == 'sales']
            ax = ax_list[i]

            # sales plot
            ax.plot(
                sku_orders['order_date'],
                sku_orders['order_quantity'],
                color=colors[i],
                label=sku,
                linestyle='-'
            )

            # forecast plot
            ax.plot(
                sku_forecast['order_date'],
                sku_forecast['order_quantity'],
                color=colors[i],
                label=sku,
                linestyle='--'
            )

            ax.set_title(sku)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            for xc in xcoords:
                ax.axvline(x=datetime.strptime(xc, "%Y-%m-%d"), color='black', linestyle='--')

        plt.plot()


if __name__ == '__main__':
    dog = DogBandana()
    dog.read_orders('../data/orders.csv')
    dog.impute_missing_sales()
    dog.shift_dates()
    dog.normalize_products()
    dog.smooth()
    dog.forecast(weeks=53)
    dog_results = dog.combine_both()
    dog_results.to_csv('../data/forecast_BANDANA014.csv', index=False)
    dog.plot_forecast(dog_results)
    # dog.print_orders()
