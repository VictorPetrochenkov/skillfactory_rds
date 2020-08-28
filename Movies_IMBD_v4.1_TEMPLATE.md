```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import collections
```


```python
data = pd.read_csv('movie_bd_v5.csv')
```


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.889000e+03</td>
      <td>1.889000e+03</td>
      <td>1889.000000</td>
      <td>1889.000000</td>
      <td>1889.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.431083e+07</td>
      <td>1.553653e+08</td>
      <td>109.658549</td>
      <td>6.140762</td>
      <td>2007.860773</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.858721e+07</td>
      <td>2.146698e+08</td>
      <td>18.017041</td>
      <td>0.764763</td>
      <td>4.468841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000e+06</td>
      <td>2.033165e+06</td>
      <td>63.000000</td>
      <td>3.300000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000e+07</td>
      <td>3.456058e+07</td>
      <td>97.000000</td>
      <td>5.600000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.800000e+07</td>
      <td>8.361541e+07</td>
      <td>107.000000</td>
      <td>6.100000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.200000e+07</td>
      <td>1.782626e+08</td>
      <td>120.000000</td>
      <td>6.600000</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.800000e+08</td>
      <td>2.781506e+09</td>
      <td>214.000000</td>
      <td>8.100000</td>
      <td>2015.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Предобработка


```python
answers = {} # создадим словарь для ответов

# Форматируем дату столбца release_date
data['release_date'] = pd.to_datetime(data.release_date) 

# Выводим процентный список пропущенных данных
print('Процентный список пропущенных данных')
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
    
# Формируем в таблице новый столбец profit, в который занесены значения прибыли 
# по каждому фильму (разность между кассовой выручкой и бюджетом) 
data['profit'] =  data['revenue'] - data['budget']

# Чистим датасет от строк где есть некорректные записи
data[data.imdb_id != 'tt0884732']

# Получаем сводную информацию по датасету data
print('-----------------\n')
print('Сводная информацию по датасету data')
data.info()

# Создаём функцию преобразования списка в словарь, служащей для формирования словаря с уникальными ключами
# из списка в котором есть повторяющиеся элементы
def list_to_dict(list_elements):
    list_elements = list(list_elements)
    c = collections.Counter()
    for category in list_elements:
        c[category] += 1
    return c
```

    Процентный список пропущенных данных
    imdb_id - 0.0%
    budget - 0.0%
    revenue - 0.0%
    original_title - 0.0%
    cast - 0.0%
    director - 0.0%
    tagline - 0.0%
    overview - 0.0%
    runtime - 0.0%
    genres - 0.0%
    production_companies - 0.0%
    release_date - 0.0%
    vote_average - 0.0%
    release_year - 0.0%
    -----------------
    
    Сводная информацию по датасету data
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1889 entries, 0 to 1888
    Data columns (total 15 columns):
     #   Column                Non-Null Count  Dtype         
    ---  ------                --------------  -----         
     0   imdb_id               1889 non-null   object        
     1   budget                1889 non-null   int64         
     2   revenue               1889 non-null   int64         
     3   original_title        1889 non-null   object        
     4   cast                  1889 non-null   object        
     5   director              1889 non-null   object        
     6   tagline               1889 non-null   object        
     7   overview              1889 non-null   object        
     8   runtime               1889 non-null   int64         
     9   genres                1889 non-null   object        
     10  production_companies  1889 non-null   object        
     11  release_date          1889 non-null   datetime64[ns]
     12  vote_average          1889 non-null   float64       
     13  release_year          1889 non-null   int64         
     14  profit                1889 non-null   int64         
    dtypes: datetime64[ns](1), float64(1), int64(5), object(8)
    memory usage: 221.5+ KB
    

# 1. У какого фильма из списка самый большой бюджет?


```python
# Определяем наименование фильма 
NameFilm = data.loc[data['budget'] == data['budget'].max()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_id = data.loc[data['budget'] == data['budget'].max()]['imdb_id']

print(NameFilm_id)
```

    723    Pirates of the Caribbean: On Stranger Tides
    Name: original_title, dtype: object
    723    tt1298650
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 1
answers['1'] = 'Pirates of the Caribbean: On Stranger Tides (tt1298650)'
```

# 2. Какой из фильмов самый длительный (в минутах)?


```python
# Определяем наименование фильма 
NameFilm = data.loc[data['runtime'] == data['runtime'].max()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_runtme = data.loc[data['runtime'] == data['runtime'].max()]['imdb_id']
print(NameFilm_runtme)
```

    1157    Gods and Generals
    Name: original_title, dtype: object
    1157    tt0279111
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 2
answers['2'] = 'Gods and Generals (tt0279111)'
```

# 3. Какой из фильмов самый короткий (в минутах)?






```python
# Определяем наименование фильма 
NameFilm = data.loc[data['runtime'] == data['runtime'].min()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_runtme = data.loc[data['runtime'] == data['runtime'].min()]['imdb_id']
print(NameFilm_runtme)
```

    768    Winnie the Pooh
    Name: original_title, dtype: object
    768    tt1449283
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 3
answers['3'] = 'Winnie the Pooh (tt1449283)'
```

# 4. Какова средняя длительность фильмов?



```python
# Определяем среднеарифметическое значение по все коллонке runtime, в которой находятся значения длительности фильмов в минутах.
# Производится округление до ближайшего целого
round(data['runtime'].mean())
```




    110




```python
# Фиксируем ответ на вопрос № 4 путём округления до ближайшего целого
answers['4'] = '110'
```

# 5. Каково медианное значение длительности фильмов? 


```python
# Определяем медиану коллонки runtime, в которой находятся значения длительности фильмов в минутах
# Производится округление до ближайшего целого
round(data['runtime'].median())
```




    107




```python
# Фиксируем ответ на вопрос № 5
answers['5'] = '107'
```

# 6. Какой самый прибыльный фильм?



```python
# Определяем наименование фильма 
NameFilm = data.loc[data['profit'] == data['profit'].max()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_profit = data.loc[data['profit'] == data['profit'].max()]['imdb_id']
print(NameFilm_profit)
```

    239    Avatar
    Name: original_title, dtype: object
    239    tt0499549
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 6
answers['6'] = 'Avatar (tt1449283)'
```

# 7. Какой фильм самый убыточный? 


```python
# Определяем наименование фильма 
NameFilm = data.loc[data['profit'] == data['profit'].min()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_profit = data.loc[data['profit'] == data['profit'].min()]['imdb_id']
print(NameFilm_profit)
```

    1245    The Lone Ranger
    Name: original_title, dtype: object
    1245    tt1210819
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 7
answers['7'] = 'The Lone Ranger (tt1210819)'
```

# 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?


```python
# Из колонки profit формируем список всех значений больше нуля и подсчитываем количество элементов полученного списка
len([x for x in data['profit'] if x>0])
```




    1478




```python
# Фиксируем ответ на вопрос № 8
answers['8'] = '1478'
```

# 9. Какой фильм оказался самым кассовым в 2008 году?


```python
# Формируем новый датасет, в который входят фильмы вышедшие только в 2008 году
data_2008 = data[data.release_year == 2008]

# Определяем наименование фильма 
NameFilm_2008 = data_2008.loc[data_2008['revenue'] == data_2008['revenue'].max()]['original_title']
print(NameFilm_2008)

# Определяем id фильма
NameFilm_profit_2008 = data_2008.loc[data_2008['revenue'] == data_2008['revenue'].max()]['imdb_id']
print(NameFilm_profit_2008)
```

    599    The Dark Knight
    Name: original_title, dtype: object
    599    tt0468569
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 9
answers['9'] = 'The Dark Knight (tt0468569)'
```

# 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?



```python
# Формируем новый датасет, в который входят фильмы вышедшие в диапозоне с 2012 по 2014 годы включительно
data_2012_2014 = data.query('release_year >=2012 & release_year <=2014')

# Определяем наименование фильма 
NameFilm = data_2012_2014.loc[data_2012_2014['profit'] == data_2012_2014['profit'].min()]['original_title']
print(NameFilm)

# Определяем id фильма
NameFilm_profit = data_2012_2014.loc[data_2012_2014['profit'] == data_2012_2014['profit'].min()]['imdb_id']
print(NameFilm_profit)
```

    1245    The Lone Ranger
    Name: original_title, dtype: object
    1245    tt1210819
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 10
answers['10'] = 'The Lone Ranger (tt1210819)'
```

# 11. Какого жанра фильмов больше всего?


```python
# Формируем список всех жанров
list_genres = list(('|'.join([str(x) for x in data['genres']])).split('|'))

# Группируем список в словарь с, ключём которого является жанр, а значение количество одинаковых жанров из списка list_genres
c = list_to_dict(list_genres)

# Определяем жанр, с наибольшим значением в словаре с
x = 0
for genres, num in c.items():
    if num > x:
        x = num
        g = genres
print(g)
```

    Drama
    


```python
# Фиксируем ответ на вопрос № 11
answers['11'] = 'Drama'
```

# 12. Фильмы какого жанра чаще всего становятся прибыльными? 


```python
# Формируем новый датасет с положительным прибылью
data_profit = data[data.profit > 0]

# Формируем список всех жанров датасета data_profit
list_genres = list(('|'.join([str(x) for x in data_profit['genres']])).split('|'))

# Группируем список в словарь g, ключём которого является жанр, а значение количество одинаковых жанров из списка list_genres
c = list_to_dict(list_genres)

# Определяем жанр, с наибольшим значением в словаре c
x = 0
for genres, num in c.items():
    if num > x:
        x = num
        g = genres
print (g)
```

    Drama
    


```python
# Фиксируем ответ на вопрос № 12
answers['12'] = 'Drama'
```

# 13. У какого режиссера самые большие суммарные кассовые сбооры?


```python
# Формируем список всех директоров
list_director = list(('|'.join([str(x) for x in data['director']])).split('|'))

# Группируем список в словарь в, ключём которого является имя директора, а значение - количество одинаковых директоров из списка list_director
d = list_to_dict(list_director)
    
# Из ключей словаря d формируем список уникальных имён директоров
list_director = d.keys()

# Создаём новый словарь dict_director куда будудт вносится имена директоров - ключ и суммарная кассовая выручка от их фильмов
dict_director = {}

# В цикле формируем суммы - summa_revenue кассовых выручек по столбцу revenue нового датасетам, 
# получаемого для каждого директора из списка list_director и заносим эти данные в словарь dict_director
for x in list_director:
    summa_revenue = data[data.director.str.contains(x,na=False)]['revenue'].sum()
    dict_director.setdefault(x,summa_revenue)
    
# Определяем директора, с наибольшим значением суммарной выручки revenue словаре dict_director 
x = 0
for director, revenuet in dict_director.items():
    if revenuet > x:
        x = revenuet
        g = director
print(g)
```

    Peter Jackson
    


```python
# Фиксируем ответ на вопрос № 13
answers['13'] = 'Peter Jackson'
```

# 14. Какой режисер снял больше всего фильмов в стиле Action?


```python
t = 0 # Контрольная переменная, по максимальному значению которой будет определятся директор кортины

# В цикле перебора директоров картин из списка list_director, будет формироваться датасет - data_director с фильмами директора.
# Датасета data_director, фильтруется по столбцу genres на наличие записи Action, с последующим определением количества строк.
# Контрольной переменной присваетивается большее значение количества строк num_action в цикле и имя директора - director.
for x in list_director:
    data_director = data[data.director.str.contains(x,na=False)]
    num_action = len(data_director[data_director.genres.str.contains('Action',na=False)].index)
    if num_action > t:
        t = num_action
        director = x
print(director)
```

    Robert Rodriguez
    


```python
# Фиксируем ответ на вопрос № 14
answers['14'] = 'Robert Rodriguez'
```

# 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 


```python
# Формируем новый датасет, в который входят фильмы вышедшие только в 2012 году
data_2012 = data[data.release_year == 2012]

# Формируем список всех актёров снимавшихся в фильмах вышедших только 2012 году
list_actors_2012 = list(('|'.join([str(x) for x in data_2012['cast']])).split('|'))

# Группируем список в словарь c, ключём которого является имена актёраов, а значением - количество фильмов (дата выхода 2012 год), в которых он снимался.
c = list_to_dict(list_actors_2012)

# Оставляем в list_actors уникальные значения элементов - имена актёров без повторений
list_actors_2012 = c.keys()

t = 0 # Контрольная переменная, по максимальному значению которой будет определятся актёр

# В цикле формируем суммы - summa_revenue кассовых выручек по столбцу revenue датасета data_2012, 
# получаемого для каждого актёра из списка list_actors
# Контрольной переменной присваетивается большее значение summa_revenue в цикле и имя директора - actors.
for x in list_actors_2012:
    summa_revenue = data_2012[data_2012.cast.str.contains(x,na=False)]['revenue'].sum()
    if summa_revenue > t:
        t = summa_revenue
        actors = x
print(actors)
```

    Chris Hemsworth
    


```python
# Фиксируем ответ на вопрос № 15
answers['15'] = 'Chris Hemsworth'
```

# 16. Какой актер снялся в большем количестве высокобюджетных фильмов?


```python
# Формируем новый датасет - data_sup_budget_mean, в который входят фильмы, бюджетные затры на производство которых ушла 
# сумма выше среднего значения бюджетов всего датасета
data_sup_budget_mean = data[data.budget >= data['budget'].mean()]

# Формируем список актёров из полученного датасета
list_actors_high_budget = list(('|'.join([str(x) for x in data_sup_budget_mean['cast']])).split('|'))

# Группируем список в словарь c, ключём которого является имена актёраов, а значением - количество фильмов
# в которых они снимались
c = list_to_dict(list_actors_high_budget)

t = 0 # Контрольная переменная, по максимальному значению которой будет определятся актёр
# В цикле, по значению quantity, определяем максимальное значение фильмов, в которых снимался тот или иной актёр
for actors, quantity in c.items():
    if quantity > t:
        t = quantity
        g = actors
print(g)
```

    Matt Damon
    


```python
# Фиксируем ответ на вопрос № 16
answers['16'] = 'Matt Damon'
```

# 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 


```python
# Формируем новый датасет - data_NicolasCage, в котором присутсвуют только фильмы с участием Nicolas Cage.
data_NicolasCage = data[data.cast.str.contains('Nicolas Cage',na=False)]

# Формируем список всех жанров датасета - data_NicolasCage
list_genres_NicolasCage = list(('|'.join([str(x) for x in data_NicolasCage['genres']])).split('|'))

# Группируем список list_genres_NicolasCage в словарь c, ключём которого является жанр фильма, а значением количество фильмов 
# этого жанра, в которых снимался Nicolas Cage
c = list_to_dict(list_genres_NicolasCage)

t = 0 # Контрольная переменная, по максимальному значению которой будет определятся актёр
# В цикле, по значению quantity, определяем максимальное значение жанров.
for genres, quantity in c.items():
    if quantity > t:
        t = quantity
        g = genres
print(g)
```

    Action
    


```python
# Фиксируем ответ на вопрос № 17
answers['17'] = 'Action'
```

# 18. Самый убыточный фильм от Paramount Pictures


```python
# Определяем из нового датасета, в которм присутсвют фильмы снимаемые при участии Paramount Pictures минимальное значения прибыли
min_profit_ParamountPictures = data[data.production_companies.str.contains('Paramount Pictures',na=False)]['profit'].min()

# Формируем датасет data_min, накладывая фильтр на столбец profit min_profit_ParamountPictures
data_min= data[data.profit == min_profit_ParamountPictures]

# Выводим столбцы из data_min с названием фильмов и id фильмов
print(data_min['original_title'])
print(data_min['imdb_id'])

# Даннае решение плставленной задачи позволяет вывести все фильмы, так как минимальная убыточность может быть одинаковой 
# для нескольких фильмов в анализируемом датасете data.
```

    925    K-19: The Widowmaker
    Name: original_title, dtype: object
    925    tt0267626
    Name: imdb_id, dtype: object
    


```python
# Фиксируем ответ на вопрос № 18
answers['18'] = 'K-19: The Widowmaker (tt0267626)'
```

# 19. Какой год стал самым успешным по суммарным кассовым сборам?


```python
# Формируем список всех лет в которых выходили фильмы
list_yeas =[x for x in data['release_year']]

# Сводим фильмы в словарь y, ключём которого является год, а значением, количество фильмов вышедших в этот год
y = list_to_dict(list_yeas)

# Получаем сортированный список уникальных значений годов.
list_yeas = sorted(y.keys())

t = 0 # Контрольная переменная, по максимальному значению которой будет определятся суммарный кассовый сбор

# В цикле перебора годов, формируются датасеты по каждому году отдельно и производится суммирование всех значений столбца revenue - кассовый сбор.
for x in list_yeas:
    suma_yeas = data[data.release_year == x]['revenue'].sum()
    if suma_yeas > t:
        t = suma_yeas
        yeas = x
print(yeas)
```

    2015
    


```python
# Фиксируем ответ на вопрос № 19
answers['19'] = '2015'
```

# 20. Какой самый прибыльный год для студии Warner Bros?


```python
# Формируем новый датасет data_WarnerBros, в котором присутсвуют все фильмы снятые при участии студии Warner Bros
data_WarnerBros = data[data.production_companies.str.contains('Warner Bros',na=False)]

t = 0 # Контрольная переменная, по максимальному значению которой будет определятся прибыльный год

# В цикле перебора по годам, формируется датасет из data_WarnerBros для этого года, и по столбцу profit определяется общая сумма прибыли.
for x in list_yeas:
    suma_yeas_WarnerBros = data_WarnerBros[data_WarnerBros.release_year == x]['profit'].sum()
    if suma_yeas_WarnerBros > t:
        t = suma_yeas_WarnerBros
        yeas = x
print(yeas)
```

    2014
    


```python
# Фиксируем ответ на вопрос № 20
answers['20'] = '2014'
```

# 21. В каком месяце за все годы суммарно вышло больше всего фильмов?


```python
list_month =  [i for i in range(1,13)] # Создаём список номеров 12 месяцев
t = 0 # Контрольная переменная, по максимальному значению которой ,будет определятся искомый месяц

# В цикле перебора по месяцам из датасета data, формируется датасет отфильтрованны по столбцу release_date и определяется количество строк.
for x in list_month:
    num_month = len(data[data.release_date.dt.month == x]) # dt.month - формирует из полной даты номер месяца, от 1 до 12
    if num_month > t:
        t = num_month
        month = x
# Расшифровка результата по названию месяца
if month == 1:
	month = 'Январь'
elif month == 2:
	month = 'Февраль'
elif month == 3:
	month = 'Март'
elif month == 4:
	month = 'Апрель'
elif month == 5:
	month = 'Май'
elif month == 6:
	month = 'Июнь'
elif month == 7:
	month = 'Июль'
elif month == 8:
	month = 'Август'
elif month == 9:
	month = 'Сентябрь'
elif month == 10:
	month = 'Октябрь'
elif month == 11:
	month = 'Ноябрь'
elif month == 12:
	month = 'Декабрь'

print(month)
```

    Сентябрь
    


```python
# Фиксируем ответ на вопрос № 21
answers['21'] = 'Сентябрь'
```

# 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)


```python
suma = 0 # Начальное значение суммы фильмов

# В цикле перебора трёх летних месяцев 6, 7 и 8, определяется размер датасетов по каждому месяцу.
for x in range(6,9):
    suma+= len(data[data.release_date.dt.month == x])
print(suma)
```

    450
    


```python
# Фиксируем ответ на вопрос № 22
answers['22'] = '450'
```

# 23. Для какого режиссёра зима – самое продуктивное время года? 


```python
# Формируем три датасета для каждого зимнего месяца
data_winter_12 = data[data.release_date.dt.month == 12]
data_winter_1 = data[data.release_date.dt.month == 1]
data_winter_2 = data[data.release_date.dt.month == 2]

# Объединяем data_winter_12, data_winter_1, data_winter_2 в единый датасет data_summer
data_winter = pd.concat([data_winter_12, data_winter_1, data_winter_2])

# Формируем список всех режиссёров фильмы которых выходили только в зимние месяцы
list_director_winter = list(('|'.join([str(x) for x in data_winter['director']])).split('|'))

# Сводим список list_director_winter в словарь где ключом является имя режиссёра - director, а значение - количество фильмов вышедших зимой. 
c = list_to_dict(list_director_winter)

t = 0 # Контрольная переменная, по максимальному значению которой, будет определятся режиссёр director.

# В цикле из словаря вбирается ключ - режиссёр и значение - количество фильмов
for x, num_movies in c.items():
    if num_movies > t:
        t = num_movies
        director = x
print(director)
```

    Peter Jackson
    


```python
# Фиксируем ответ на вопрос № 23
answers['23'] = 'Peter Jackson'
```

# 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?


```python
# Формируем список всех киностудий из датасета data
list_film_studio = list(('|'.join([str(x) for x in data['production_companies']])).split('|'))

# Формируем словарь с уникальными ключами - название киностудии и значением - количество фильмов снятых при участии этих компаний
f = list_to_dict(list_film_studio)

# Формируем список киностудий по ключам словаря f
list_film_studio = f.keys()

t = 0 # Контрольная переменная, по максимальному значению которой, будет определятся самое длинное название фильма

# В цикле перебора наименований киностудий формируются датасеты - data_movies,которые содержат наименования фильмов снятыми при их участии
# Из этих датасетов выбираются наименования фильмов, определяется количество символов в этих наименованиях
# Если список не пустой, то подсяитывается среднеарифитическое значение длины наименований фильмов по каждой киностудии
for x in list_film_studio:
    data_movies = data[data.production_companies.str.contains(x, na=False)]    
    movies = [len(str(y)) for y in data_movies['original_title'] if len(y) != 0]
    if len(movies) != 0:
        mean = sum(movies)/len(movies)
        if mean > t:
            t = mean
            companies = x
print(companies)
```

    Four By Two Productions
    


```python
# Фиксируем ответ на вопрос № 24
answers['24'] = 'Four By Two Productions'
```

# 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?


```python
t = 0 # Контрольная переменная, по максимальному значению которой, будет определятся киностудия

# В цикле перебора наименований киностудий формируются датасеты -data_movies фильмы которых снятs при участии этой киностудии
# Из датасета data_movies выбираются описания количество пробелов  заносится в список movie_description. Количество пробелов
# определяет количество слов в тексте +1
# Если список не пустой, то по каждому списку определяется среднеарифметическое значение для всех описаний фильмов снятых киностудией.
# Переменной t присваивается текущее максимальному значение mean
for x in list_film_studio:
    data_movies = data[data.production_companies.str.contains(x, na=False)]
    movie_description = list([y.count(' ') for y in data_movies['overview'] if str(y) != ''])
    if len(movie_description) != 0:
        mean = (sum(movie_description)+len(movie_description)) / len(movie_description)
        if mean > t:
            t = mean
            companies = x
print(companies)
```

    Midnight Picture Show
    


```python
# Фиксируем ответ на вопрос № 25
answers['25'] = 'Midnight Picture Show'
```

# 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
по vote_average


```python
# Определяем 1% фильмов
pr = round(len(data)/100)

# Формируем полный список названий фильмов
list_original_title = [x for x in data['original_title']]

# Создаём словарь в котором будет храниться название фильма  original_title - ключ и значение рейтинга vote_average
dict_original_title_vote_average = {}

# Заполняем словарь в цикле по каждому фильму (исключаем возможные ошибки встречающиеся в столбцу vote_average)
for x in list_original_title:
    try:
        va = data[data.original_title.str.contains(x,na=False)].vote_average.unique()[0]
    except:
        va = 0
    if va != 0:
        dict_original_title_vote_average.setdefault(x,va)
# Сортируем словарь по по значению рейтинга (по возрастающей)
list_d = list(dict_original_title_vote_average.items())
list_d.sort(key=lambda i: i[1])

# Осуществляем реверс словаря и выводим первые 1% фильмов, с самым высоким рейтингом.
list_d.reverse()
a = list_d[:pr]
print(a)
```

    [('The Dark Knight', 8.1), ('The Imitation Game', 8.0), ('Interstellar', 8.0), ('Room', 8.0), ('Inside Out', 8.0), ('Memento', 7.9), ('The Return', 7.9), ('12 Years a Slave', 7.9), ('The Wolf of Wall Street', 7.9), ('The Lord of the Rings: The Return of the King', 7.9), ('The Pianist', 7.9), ('Inception', 7.9), ('The Grand Budapest Hotel', 7.9), ('Gone Girl', 7.9), ('Guardians of the Galaxy', 7.9), ('There Will Be Blood', 7.8), ('Eternal Sunshine of the Spotless Mind', 7.8), ('The Prestige', 7.8), ('Dallas Buyers Club', 7.8)]
    


```python
# Фиксируем ответ на вопрос № 26
answers['26'] = 'Inside Out, The Dark Knight, 12 Years a Slave'
```

# 27. Какие актеры чаще всего снимаются в одном фильме вместе?



```python
# Формируем полный список актёров снявшихся в фильмах
list_actors = list(('|'.join([str(x) for x in data['cast']])).split('|'))
c = list_to_dict(list_actors)
list_actors = c.keys()

# Чистим список имён актёров от символа '.
list_actors_changes = [i.replace("'",' ') for i in list_actors]

# Формируем групповой список актёров снявшихся в фильмах и чистим от символа '
list_cast = [x.replace("'",' ') for x in data['cast']]

# Символ ' будет мешать корректной работе метода find()

t = 7 # Контрольная переменная, по максимальному значению которой бедте определятся актёрский дуэт снявшихся в большем количестве фильмов.

# В цикле по каждому актёру из списка - list_cast удаляются все элементы в которых он отсутствует. 
for actor_x in list_actors_changes:
    cast_actor_x = [x for x in list_cast if x.find(actor_x) >=0]
    
    # В цикле по каждому актёру из списка - cast_actor_x удаляются все элементы в которых он отсутствует.
    # Контрольной переменной присваивается тот список количество оставшихся элементов больше или равно предыдущим
    for actor_y in list_actors_changes:
        if actor_x != actor_y:
            cast_actor_y = [x for x in cast_actor_x if x.find(actor_y) >=0]
            if len(cast_actor_y) >= t:
                t = len(cast_actor_y)
                print(t,'=' ,actor_x,'&', actor_y)
                
# Результат необходимо выводить весь (чтобы сократить список, значение контрольной переменной взято равным 7), так как есть актёрские дуэты с одинаковым количеством фильмов. Условие len(cast_actor_y) > t
# не выведет другие актёрские дуэты с таким же количеством фильмов.
```

    8 = Daniel Radcliffe & Emma Watson
    8 = Daniel Radcliffe & Rupert Grint
    8 = Emma Watson & Daniel Radcliffe
    8 = Emma Watson & Rupert Grint
    8 = Rupert Grint & Daniel Radcliffe
    8 = Rupert Grint & Emma Watson
    


```python
# Фиксируем ответ на вопрос № 27
answers['27'] = 'Daniel Radcliffe & Rupert Grint'
```

# Submission


```python
# в конце можно посмотреть свои ответы к каждому вопросу
answers
```




    {'1': 'Pirates of the Caribbean: On Stranger Tides (tt1298650)',
     '2': 'Gods and Generals (tt0279111)',
     '3': 'Winnie the Pooh (tt1449283)',
     '4': '110',
     '5': '107',
     '6': 'Avatar (tt1449283)',
     '7': 'The Lone Ranger (tt1210819)',
     '8': '1478',
     '9': 'The Dark Knight (tt0468569)',
     '10': 'The Lone Ranger (tt1210819)',
     '11': 'Drama',
     '12': 'Drama',
     '13': 'Peter Jackson',
     '14': 'Robert Rodriguez',
     '15': 'Chris Hemsworth',
     '16': 'Matt Damon',
     '17': 'Action',
     '18': 'K-19: The Widowmaker (tt0267626)',
     '19': '2015',
     '20': '2014',
     '21': 'Сентябрь',
     '22': '450',
     '23': 'Peter Jackson',
     '24': 'Four By Two Productions',
     '25': 'Midnight Picture Show',
     '26': 'Inside Out, The Dark Knight, 12 Years a Slave',
     '27': 'Daniel Radcliffe & Rupert Grint'}




```python
# и убедиться что ни чего не пропустил)
len(answers)
```




    27


