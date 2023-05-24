using MQTTnet;
using MQTTnet.Client;
using MQTTnet.Protocol;
using System;
using System.IO;
using System.Threading.Tasks;

namespace MqttPublisher
{
    class Program
    {
        static async Task Main(string[] args)
        {

            // Set up MQTT client options
            var mqttOptions = new MqttClientOptionsBuilder()
                .WithTcpServer("localhost", 1883)
                .WithClientId("mqtt_publisher")
                .Build();

            // Create MQTT client
            var mqttClient = new MqttFactory().CreateMqttClient();

            // Connect to MQTT broker
            await mqttClient.ConnectAsync(mqttOptions);

            // Open text file with sensor readings
            //var filePath = @"C:\Users\User\Desktop\Python\Wind_Turbine_Gearbox\MLR\new_data_entry_realtime.txt";
            //var filePath = @"C:\\Users\\User\\Desktop\\Python\\Wind_Turbine_Gearbox\\MLR\\T06_2017\\T06_2017_For_telegraf.txt";
            var filePath = @"C:\\Users\\User\\Desktop\\Python\\Wind_Turbine_Gearbox\\MLR\\T06_2017\\T06_2017_For_telegraf_ts.txt";
            var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            var reader = new StreamReader(fileStream);

            // Read and publish sensor readings line by line
            var lineCount = 0;
            var startTime = DateTime.Now;

        
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                

                // Get current time in Unix nanosecond format
                long unixNanoSeconds = (long)(DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds * 1_000_000_000;

                // Create MQTT message with sensor reading
                var message = new MqttApplicationMessageBuilder()
                    .WithTopic("sensor/readings")
                    //.WithPayload(line)
                    .WithPayload(line+" "+unixNanoSeconds)
                    .WithQualityOfServiceLevel(MqttQualityOfServiceLevel.AtMostOnce)
                    .WithRetainFlag(false)
                    .Build();

                // Publish MQTT message
                await mqttClient.PublishAsync(message);

                // Print MQTT message to console
                //Console.WriteLine($"Published: {DateTime.Now},{line}");

                // Print MQTT message to console
                Console.WriteLine($"Published: {line+" "+unixNanoSeconds}");

                // Print MQTT message to console with Unix nanosecond formatted time
                //Console.WriteLine($"Published: {unixNanoSeconds},{line}");

                // Increment line count
                lineCount++;

                // Wait for one second before publishing the next message
                await Task.Delay(1000);

            }

            // Close text file and disconnect from MQTT broker
            reader.Close();
            fileStream.Close();
            await mqttClient.DisconnectAsync();
            Console.WriteLine("****End of Dataset reached, connection disabled!!!****");
        }
    }
}

