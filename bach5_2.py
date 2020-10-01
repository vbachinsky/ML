import pandas as pd
import numpy as np


def set_prnt_option():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = '{:,.2f}'.format

def convert_text_to_binary(text):
    if text == "y":
        return 1
    elif text == "n":
        return 0
    else:
        return None

def convert_rating_to_grade(rating):
    rating_mapping = {"excellent": 5, "good": 4, "ok": 3, "bad": 2, np.NaN: 0}
    if rating in rating_mapping:
        return rating_mapping[rating]
    else:
        return None

def convert_age_to_range(age):
    if 6 <= age < 18:
        return "6-18"
    elif 18 <= age < 35:
        return "18-35"
    elif 35 <= age < 99:
        return "35-99"
    else:
        return "unknown"

set_prnt_option()
conversion_df = pd.read_csv('data/ad_conversion.csv')


# data preparation
conversion_df['last name'].fillna("", inplace=True)

conversion_df.at[8, "gender"] = "F"
conversion_df.at[14, "gender"] = "M"
conversion_df.at[29, "gender"] = "F"
conversion_df.at[37, "gender"] = "M"

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] > 100:
        conversion_df.at[z, "seen count"] = 0

conversion_df.insert(conversion_df.columns.get_loc("gender"), "full name", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "full name"] = (conversion_df.at[z, "first name"] + ' ' + conversion_df.at[z, "last name"]).strip()

conversion_df.drop(columns=["first name", "last name"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "birthday", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "birthday"] = pd.Timestamp(day=conversion_df.at[z, "day of birth"],
                                                month=conversion_df.at[z, "month of birth"],
                                                year=conversion_df.at[z, "year of birth"])

conversion_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)


for z in range(len(conversion_df)):
    conversion_df.at[z, "followed ad"] = convert_text_to_binary((conversion_df.at[z, "followed ad"]))
    conversion_df.at[z, "made purchase"] = convert_text_to_binary((conversion_df.at[z, "made purchase"]))


for z in range(len(conversion_df)):
    conversion_df.at[z, "user rating"] = convert_rating_to_grade(conversion_df.at[z, "user rating"])

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "age", None)
for z in range(len(conversion_df)):
    conversion_df.at[z, "age"] = (pd.Timestamp("01-01-2019") - conversion_df.at[z, "birthday"]).days // 365

conversion_df.drop(columns=["birthday"], inplace=True)
conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "age bucket", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "age bucket"] = convert_age_to_range(conversion_df.at[z, "age"])

conversion_df.drop(columns=["age"], inplace=True)
conversion_df = conversion_df.astype({"followed ad": "int64", "made purchase": "int64", "user rating": "int64"})

# creation of new criteria
conversion_df.insert(conversion_df.columns.get_loc("user rating"), "ad effectiveness", None)

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] == 0:
        conversion_df.at[z, "ad effectiveness"] = -1
    else:
        conversion_df.at[z, "ad effectiveness"] = (conversion_df.at[z, "followed ad"] +
                                                   conversion_df.at[z, "made purchase"]) / (2 * conversion_df.at[z, "seen count"])

conversion_df = conversion_df.astype({"ad effectiveness": "float64"})
print(conversion_df)


# Colors grouped
colors_grouped = (conversion_df[["color scheme", "followed ad", "made purchase"]]).groupby("color scheme")
print('\n Grouped by color')

for i in colors_grouped:
    print(i)

color = set()
for z in range(len(conversion_df)):
    color.add(conversion_df.at[z, "color scheme"])

color_dict = dict.fromkeys(color, 0)
color_vs_followed = dict.fromkeys(color, 0)
color_vs_purchase = dict.fromkeys(color, 0)

for z in color:
    for y in range(len(conversion_df)):
        if conversion_df.at[y, "color scheme"] == z:
            color_dict[z] += 1
            color_vs_followed[z] += conversion_df.at[y, "followed ad"]
            color_vs_purchase[z] += conversion_df.at[y, "made purchase"]
for z in color:
    print('For color {}: visites - {}, followed - {}, made purchase - {}'. format(z, color_dict[z], color_vs_followed[z], color_vs_purchase[z]))


# Gender grouped
gender_grouped = (conversion_df[["gender", "seen count", "made purchase"]]).groupby("gender")
print('\n Grouped by gender')

for i in gender_grouped:
    print(i)

gender = set()
for z in range(len(conversion_df)):
    gender.add(conversion_df.at[z, "gender"])

gender_dict = dict.fromkeys(gender, 0)
gender_vs_seen = dict.fromkeys(gender, 0)
gender_vs_purchase = dict.fromkeys(gender, 0)

for z in gender:
    for y in range(len(conversion_df)):
        if conversion_df.at[y, "gender"] == z:
            gender_dict[z] += 1
            gender_vs_seen[z] += conversion_df.at[y, "seen count"]
            gender_vs_purchase[z] += conversion_df.at[y, "made purchase"]
for z in gender:
    print('For sex {}: visites - {}, seen - {}, made purchase - {}'.format(z, gender_dict[z], gender_vs_seen[z], gender_vs_purchase[z]))


# Age grouped
age_grouped = (conversion_df[["age bucket", "color scheme", "seen count", "made purchase"]]).groupby("age bucket")
print('\n Grouped by age')

for i in age_grouped:
    print(i)

age = set()
for z in range(len(conversion_df)):
    age.add(conversion_df.at[z, "age bucket"])

age_dict = dict.fromkeys(age, 0)
age_vs_seen = dict.fromkeys(age, 0)
age_vs_purchase = dict.fromkeys(age, 0)
age_vs_color_and_visitors = dict.fromkeys(age, 0)
age_vs_color_and_purchases = dict.fromkeys(age, 0)

for z in age:
    temp_dict_color_1 = dict.fromkeys(color, 0)
    temp_dict_color_2 = dict.fromkeys(color, 0)
    for y in range(len(conversion_df)):
        if conversion_df.at[y, "age bucket"] == z:
            age_dict[z] += 1
            age_vs_seen[z] += conversion_df.at[y, "seen count"]
            age_vs_purchase[z] += conversion_df.at[y, "made purchase"]
    for x in color:
        for y in range(len(conversion_df)):
            if (conversion_df.at[y, "color scheme"] == x) and (conversion_df.at[y, "age bucket"] == z):
                temp_dict_color_1[x] += 1
                temp_dict_color_2[x] += conversion_df.at[y, "made purchase"]
    age_vs_color_and_visitors[z] = temp_dict_color_1
    age_vs_color_and_purchases[z] = temp_dict_color_2

for z in age:
    print('For age {}: visitors - {}, seen - {}, made purchase - {}'.format(z,
                                                                            age_dict[z],
                                                                            age_vs_seen[z],
                                                                            age_vs_purchase[z]))

for z in age:
    print('\nAge {}:'.format(z))
    for y in color:
        print('for color scheme {} are {} visitors and {} purchases'.format(y,
                                                                            age_vs_color_and_visitors[z][y],
                                                                            age_vs_color_and_purchases[z][y]))

conversion_df.to_csv(("data/prepared_data.csv"))