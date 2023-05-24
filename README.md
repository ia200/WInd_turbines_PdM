# WInd_turbines_PdM
This work implements an IoT solution that uses MQTT to publish sensor readings which is subscribed by a telegraf agent using an API that writes the data to an influx timeseriesdatabase.
The data is also used by a machine learning model that predicts the failure of the gearbox compoenent of the windturbine based on its temperature.
The data which is sent in real-time can be visualized using dashboards using influx query, grafana and a web application.
The web application was written using a flask app that runs the ML models and renders the generated plots to a HTML page along with predicted failure times as alerts.
The model was able to predict failure with some accuracy, atleast 1 day before it happens.
