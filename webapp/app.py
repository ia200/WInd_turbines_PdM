from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from influxdb_client import InfluxDBClient
from matplotlib.dates import date2num


import time
start = time.time()



my_token = "K9yF5FO1xXzr7z94Cb1TwBjtHZTejHoGz8fNtBsRXYQUVVwo3nU9UP__a6itxyVk0A9M7lgJAz70JQ0b_WiTuA=="
my_org = "Hariscope"
bucket = "WT_Scada"

#client = InfluxDBClient(url="http://localhost:8086", token=my_token, )

# Create a connection to InfluxDB
client = InfluxDBClient(
    url="http://localhost:8086",
    token="K9yF5FO1xXzr7z94Cb1TwBjtHZTejHoGz8fNtBsRXYQUVVwo3nU9UP__a6itxyVk0A9M7lgJAz70JQ0b_WiTuA==",
    org="Hariscope",
    bucket="WT_Scada",
)

query_api = client.query_api()

# Define your InfluxDB query
# Query the data from InfluxDB
query = """from(bucket: "WT_Scada")
  |> range(start: 2016-01-01T00:00:00Z, stop: 2017-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "T06")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield()
"""

# Execute the query
results = client.query_api().query_data_frame(query)

# Convert the result to a Pandas DataFrame
data = pd.DataFrame(results)

# Close the InfluxDB connection
client.close()

# Print the DataFrame
#print(df.head())

#print(data['Amb_Temp_Avg'])

column_names = data.columns.tolist()
#print(column_names)

datetime_obj = pd.to_datetime(data['_time'])

data['timestamp'] = datetime_obj.dt.strftime('%Y-%m-%d %H:%M:%S')

#WILL COME BACK TO YOU

# Remove instances where turbine power is zero or less but wind speed is above cut-in speed
data = data[(data['Amb_WindSpeed_Max'] >= 3.5) & (data['Grd_Prod_Pwr_Avg'] > 0)]

# Remove samples where at least one input or output is missing
df = data.dropna()

# Split the data into training and testing sets
train = df[:int(0.7*len(data))]
test = df[int(0.7*len(data)):]

# Define the input and output variables
input_vars = ['Nac_Temp_Avg', 'Rtr_RPM_Avg', 'Prod_LatestAvg_TotActPwr', 'Amb_Temp_Avg', 'Gear_Oil_Temp_Avg']
output_var = 'Gear_Bear_Temp_Avg'


#print(train['timestamp'])
#print(test['timestamp'])


# Train the regression model
model = LinearRegression()
model.fit(train[input_vars], train[output_var])

# Predict on the testing set
#test['predicted'] = model.predict(test[input_vars])

test.loc[:, 'predicted'] = model.predict(test[input_vars])

# Calculate RMSE, MAE, and MAPE
rmse = np.sqrt(mean_squared_error(test[output_var], test['predicted']))
mae = mean_absolute_error(test[output_var], test['predicted'])
mape = np.mean(np.abs((test[output_var] - test['predicted']) / test[output_var])) * 100
r2 = r2_score(test[output_var], test['predicted'])

metrics_list = []
metrics_list.append(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, R-Squared: {r2:.2f}')

# Print the evaluation metrics
print('RMSE:', round(rmse, 2))
print('MAE:', round(mae, 2))
print('MAPE:', round(mape, 2))
print('R^2 Score:', round(r2, 2))

#plt.legend()

# Save the plot as a PNG file
#plt.savefig('C:\\Users\\User\\Desktop\\Python\\Wind_Turbine_Gearbox\\MLR\\plot_results\\T06_2017_Jan_predictions.png')

#plt.show()


##starting code for my flask app here
app = Flask(__name__, template_folder='templates')

#prediction plot here
def generate_plot1():

    # Filter data between 11th and 18th October 2016
    start_date = '2016-10-11 00:00'
    end_date = '2016-10-18 23:50'
    mask = (test['timestamp'] >= start_date) & (test['timestamp'] <= end_date)
    test_filtered = test.loc[mask]

    # Convert timestamp column to datetime type
    test_filtered['timestamp'] = pd.to_datetime(test_filtered['timestamp'])
    # Plot the actual vs predicted values for filtered data
    plt.figure(figsize=(10,5))
    plt.plot(test_filtered['timestamp'], test_filtered[output_var], label='Actual')
    plt.plot(test_filtered['timestamp'], test_filtered['predicted'], label='Predicted', linewidth=1)
    plt.xlabel('Timestamp')
    plt.ylabel('Gearbox Temperature (Avg)')
    plt.title('Actual vs Predicted Gearbox Temperature (11th-18th Oct 2016)')
    plt.legend()
    
    # Save the plot to a BytesIO object
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    
    # Convert the plot to base64 string
    plot_data1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    
    return plot_data1


def generate_plot2():

    ###!!!CODE FOR spc chart
    # Filter data between 11th and 18th October 2017
    start_date = '2016-10-11 00:00'
    end_date = '2016-10-18 00:00'
    mask = (test['timestamp'] >= start_date) & (test['timestamp'] <= end_date)
    test_filtered = test.loc[mask]

    # Calculate the residual errors
    deviations = test_filtered[output_var] - test_filtered['predicted']

    mr = abs(deviations - deviations.shift(1))

    sigma = mr.mean() / 1.128

    ucl = 6 * sigma
    lcl = -6 * sigma

    within_limits = (deviations >= lcl) & (deviations <= ucl)

    # Convert timestamp column to datetime type
    test_filtered['timestamp'] = pd.to_datetime(test_filtered['timestamp'])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_filtered['timestamp'], deviations, label='Deviations')
    ax.plot(test_filtered['timestamp'], [ucl] * len(test_filtered), label='UCL', linestyle='--', color='red')
    ax.plot(test_filtered['timestamp'], [lcl] * len(test_filtered), label='LCL', linestyle='--', color='red')
    ax.fill_between(test_filtered['timestamp'], lcl, ucl, alpha=0.1, color='gray')
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Deviation')
    ax.set_title('Shewhart Control Chart')
    
    # Save the plot to a BytesIO object
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    
    # Convert the plot to base64 string
    plot_data2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    
    return plot_data2


def generate_plot3():

    #HERE WE FIND THE DOWNTIME BY ADDING CODE TO GET THE MAINTENANCE TIME OF GEARBOX

    # Convert timestamp column to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data.set_index('timestamp', inplace=True)

    # Filter data for 2017-10-12 to 13
    start_date = '2016-10-12 00:00:00'
    end_date = '2016-10-13 23:59:59'
    df_filtered = data.loc[start_date:end_date]

    # Convert the datetime string to a numerical value
    x_value = date2num(pd.to_datetime('2016-10-13 00:30:00'))

    failure_time = pd.to_datetime('2016-10-13 00:30:00')

    # Find the time of maintenance
    lowest_temp_since_failure = df_filtered[df_filtered.index >= failure_time]['Gear_Bear_Temp_Avg'].min()
    maintenance_time_mask = (df_filtered['Gear_Bear_Temp_Avg'] == lowest_temp_since_failure) & (df_filtered.index > failure_time)
    maintenance_time = df_filtered[maintenance_time_mask].index[0]
    maintenance_time_num = date2num(maintenance_time)


    # Calculate the downtime duration
    downtime_hours = (maintenance_time - pd.to_datetime('2016-10-13 00:30:00')).total_seconds() / 3600

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered.index, df_filtered['Gear_Bear_Temp_Avg'], label='Gear Bearing Temperature')
    ax.axvline(x=x_value, color='r', linestyle='--', label='Gearbox Failure')
    ax.axvline(x=maintenance_time_num, color='g', linestyle='--', label='Maintenance')
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Gear Bearing Temperature on 2016-10-13')
    plt.text(maintenance_time_num, 56, f'Downtime: {downtime_hours:.2f} hours')
    
    # Save the plot to a BytesIO object
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    
    # Convert the plot to base64 string
    plot_data3 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    
    return plot_data3

alerts_list = []

def generate_plot4():

    ## TO GET ALERTS AND NEW CPC CHART FOR DAY OF FAILURE

    # Filter data between 12th and 13th October 2016
    start_date = '2016-10-12 00:00'
    end_date = '2016-10-13 23:50'
    mask = (test['timestamp'] >= start_date) & (test['timestamp'] <= end_date)
    test_filtered = test.loc[mask]

    # Calculate the residual errors
    deviations = test_filtered[output_var] - test_filtered['predicted']

    mr = abs(deviations - deviations.shift(1))

    sigma = mr.mean() / 1.128

    ucl = 6 * sigma
    lcl = -6 * sigma

    # Find the time and temperature where Gearbox temperature spiked above UCL
    spikes_above_ucl = test_filtered[test_filtered['Gear_Bear_Temp_Avg'] > 50]
    if not spikes_above_ucl.empty:
        print('Gearbox temperature spiked above UCL at:')
        for index, row in spikes_above_ucl.iterrows():
            deviation_time = row['timestamp']
            deviation_value = deviations.loc[index]


            #if deviation_value > 3:
            if row['Gear_Bear_Temp_Avg'] > 50 and deviation_value > 2.3:
                print(f'Datetime: {deviation_time}, Deviation: {deviation_value:.2f}, Temperature: {row["Gear_Bear_Temp_Avg"]:.2f}')
                alerts_list.append(f'Datetime: {deviation_time}, Deviation: {deviation_value:.2f}, Temperature: {row["Gear_Bear_Temp_Avg"]:.2f}')

    within_limits = (deviations >= lcl) & (deviations <= ucl)

    # Convert timestamp column to datetime type
    test_filtered['timestamp'] = pd.to_datetime(test_filtered['timestamp'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_filtered['timestamp'], deviations, label='Deviations')
    ax.plot(test_filtered['timestamp'], [ucl] * len(test_filtered), label='UCL', linestyle='--', color='red')
    ax.plot(test_filtered['timestamp'], [lcl] * len(test_filtered), label='LCL', linestyle='--', color='red')
    ax.fill_between(test_filtered['timestamp'], lcl, ucl, alpha=0.1, color='gray')
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    ax.set_ylabel('Deviation')
    ax.set_title('Shewhart Control Chart')
    
    # Save the plot to a BytesIO object
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    
    # Convert the plot to base64 string
    plot_data4 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    
    return plot_data4


@app.route('/')
def home():
    plot_data1 = generate_plot1()
    plot_data2 = generate_plot2()
    plot_data3 = generate_plot3()
    plot_data4 = generate_plot4()
    

    return render_template('index.html', plot_data1=plot_data1, plot_data2=plot_data2,plot_data3=plot_data3, plot_data4=plot_data4, metrics=metrics_list)

@app.route('/alerts')
def alerts():
    return render_template('alerts.html', alerts=alerts_list)


if __name__ == '__main__':
    app.run(debug=True)


