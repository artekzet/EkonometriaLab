# Projekt na zaliczenie - Ekonometria AGH - Artur Zwoliński
# Temat: Przewidywanie długości przejazdu taksówki w Nowym Jorku
# Problem - na jakie długie kursy mogą liczyć taksówkarze w NY, w zależności od różnych czynników.



## Zdecydowałem się na zrobienie projektu w zestawie Spark + Scala, ponieważ pracuję ze Scalą na codzień.
## Część projektu odpowiedzialna za statystyki jest w Pythonie

### Wymagania do odpalenia:
1. Java Development Kit 11
2. IntelliJ Idea (albo inne IDE do Scali/Javy)
3. Python w wersji 3 (do drugiej części), odpalany w linii poleceń komendą ` python3 src/main/neuralNetworkTaxi.py`

### Dane pochodzą ze strony: https://academictorrents.com/details/4f465810b86c6b793d1c7556fe3936441081992e

### Wybrałem jeden plik z dostępnych, jednak program powinien zadziałać z innym plikiem (lub nawet kilkoma) tak samo, ponieważ zbiór danych już jest oczyszczony i sformatowany. Przy innym pliku wyniki cząstkowe mogą być nieco inne.

Wywołanie komendy:
`taksowki.printSchema()`

Pozwala na uzyskanie informacji, jaki schemat mają dane:
![struktura danych](strukturaDanych.png?raw=true "dane o taksowkach")

Sprawdzam, jak godzina dnia ma się do średniej odległości:

KOD:
`taksowki.withColumn("hour_of_day", hour(col("pickup_datetime")))
.groupBy("hour_of_day")
.agg(avg("trip_distance").as("averageDistance"))
.orderBy(col("averageDistance").desc_nulls_last)
.show()`

Wynik:
`
+-----------+------------------+
|hour_of_day|     totalDistance|
+-----------+------------------+

|          6|4.0884547796270665|

|          5|3.8938216618962564|

|          4|3.3533231430501993|

|          3|3.1460860792467624|

|          7| 3.040052123561827|

...

`

Najlepsze trasy taksówkarz ma szansę załapać po 3 rano. Przed północą i po 8 rano średnia długość kursów spada poniżej 3 mil.

Chciałem teraz porównać średni dystans dla poszczególnych stref odbioru. 

Kod:

`
taksowki.withColumn("zone", col("pickup_taxizone_id"))
.groupBy("zone")
.agg(avg("trip_distance").as("averageDistance"))
.orderBy(col("averageDistance").desc_nulls_last)
.show()
`

Wyniki:

`|zone|   averageDistance|

+----+------------------+

| 132|12.773339649643193|

| 117|10.256999949614206|

| 215|   9.8249731993627|

|  10| 9.422247475539946|

| 138| 8.818354919365273|

|  16| 8.359199904603884|

...
`

Ściągnąłem plik taxi_zones.csv, z niego możemy odczytać, że dwie lokalizacje, ze średnimi kursami powyżej 10 mil, to lotnisko JFK i szpital w dzielnicy Queens.

Jednak po pogrupowaniu poprzez oba czynniki na raz, wychodzi, że przez całą dobę najlepszą lokalizacją jest port JFK:

Kod:
`taksowki.withColumn("zone", col("pickup_taxizone_id"))
.withColumn("hour_of_day", hour(col("pickup_datetime")))
.groupBy("zone", "hour_of_day")
.agg(avg("trip_distance").as("averageDistance"), count("*").as("totalTrips"))
.filter("totalTrips > 30")
.orderBy(col("averageDistance").desc_nulls_last)
.show()
`


Wyniki (top 20)


`
| 132|          1|14.269474621140061|       552|

| 132|          8|14.002966104659363|       944|

| 132|         13|13.939145170010987|       620|

| 132|         12|  13.8039410944589|       543|

| 132|         16|13.222395228195987|      1002|

| 132|         14|13.181195063771835|       569|

| 132|         23|13.075548380042239|      1282|

| 132|          7|13.073548868587562|       696|

| 132|          2|12.927889895249466|       218|

| 132|         17| 12.89747596895302|      1498|

| 132|          9|12.895684942344765|       584|

| 132|         15|12.873822783068105|       790|

| 132|         22|12.675518031674123|      1110|

| 132|         21| 12.43317052768744|      1161|

| 132|         20|12.382885807061363|      1147|

| 132|         19|12.349896561115504|      1286|

| 132|         18|12.339744876285955|      1556|

| 132|         10|12.173245196915769|       416|

| 132|          0|12.056705448966982|       825|

| 132|         11|11.760049760582584|       402|

`

Mając te wszystkie informacje, możemy pogrupować wyniki i podając kod strefy oraz godzinę przewidzieć, jak długa będzie trasa.

Kod metody:
`
def przewidywanaDlugoscTrasy(strefa: Int, godzina: Int): Unit = {
grupowaniePoGodzinieIMiejscu
.filter(s"hour_of_day = $godzina")
.filter(s"pickup_taxizone_id = $strefa")
.agg(avg("trip_distance").as("averageDistance"))
.show()
}
`

Tym sposobem wiemy, że na przykład dla strefy 132, o godzinie 7, przewidywana długość trasy przejazdu do 13 mil

Zapiszmy grupowanie po godzinie i miejscu przejazdu do pliku CSV:

`grupowaniePoGodzinieIMiejscu.write
.option("header", true)
.csv("src/main/resources/hourAndPlace.csv")`

Teraz użyjmy języka Python do stworzenia sieci neuronowej

Na początek wczytanie pliku i usunięcie wierszy, gdzie są nieznane wartości


`raw_dataset = pd.read_csv(file_name)`

`dataset = raw_dataset.copy()`

`dataset = dataset.dropna()`

Kompilowanie modelu: 
`model_podrozy.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam())`

Podział na zbiory treningowy i testowy:

`train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)`

Przeprowadzamy predykcję na naszych zbiorach (chcemy tak jak wcześniej przewidzieć dystans)

`
normalizator = tf.keras.layers.Normalization(axis=-1)
normalizator.adapt(np.array(train_features))
print(normalizator.mean.numpy())
trip_distance = np.array(train_dataset[column_to_remove])
trip_distance_normal = layers.Normalization(input_shape=[1,], axis=None)
trip_distance_normal.adapt(trip_distance)
model = tf.keras.Sequential([
    trip_distance_normal,
    layers.Dense(units=1)
])
`

Podsumowanie modelu:

`
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[[163.217  13.663]]
Model: "sequential"
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization_1 (Normalizat  (None, 1)                3         
 ion)
 dense (Dense)               (None, 1)                 2
=================================================================
Total params: 5
Trainable params: 2
Non-trainable params: 3
`

Następnie uzyjemy regresji liniowej:
`
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
history = model.fit(
    train_dataset[column_to_remove],
    test_features,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
test_results = {}
test_results['distance_model'] = model.evaluate(
    test_features[column_to_remove],
    test_labels, verbose=0)
`

I na końcu wyświetlimy wykres z danymi oraz linią predykcji:

`x = tf.linspace(0.0, 250, 251)
y = model.predict(x)
def plot_distance_prediction(x, y):
    plt.scatter(train_dataset[column_to_remove], train_dataset[column_to_remove], label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Distance')
    plt.ylabel('Val')
    plt.legend()
    plt.show()
plot_distance_prediction(x, y)
`

Niestety, nie udało mi się tak dobrać parametrów w x, żeby wykres był taki deskryptywny jak w samouczku TensorFlow (https://www.tensorflow.org/tutorials/keras/regression) 