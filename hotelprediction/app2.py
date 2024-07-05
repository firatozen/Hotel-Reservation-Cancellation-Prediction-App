##oldu galiba
##round yaptım children'ı
##denormalize prediction
##resim eklendi

import streamlit as st
import pandas as pd
import joblib
import pymongo
import os
import time
import base64
from bson import ObjectId
from bson.objectid import ObjectId
from sklearn.preprocessing import LabelEncoder
import pyperclip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit import components
import io
from datetime import datetime, timedelta
import calendar
import uuid
import math
import re

# Load the saved CatBoost model
model = joblib.load('catboost_model(val).joblib')  # Update path to your saved model

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection string
db = mongo_client["hotel_bookings"]
collection = db["bookings"]

# Maximum values for each column
max_values = {
    'hotel': 1,
    'meal': 4,
    'market_segment': 7,
    'distribution_channel': 4,
    'reserved_room_type': 8,
    'deposit_type': 3,
    'customer_type': 3,
    'year': 30,
    'month': 12,
    'day': 31,
    'lead_time': 600,
    'arrival_date_week_number': 53,
    'arrival_date_day_of_month': 31,
    'stays_in_weekend_nights': 140000,
    'stays_in_week_nights': 340000,
    'adults': 5000,
    'children': 1000,
    'babies': 100000,
    'is_repeated_guest': 10000,
    'previous_cancellations': 260000,
    'previous_bookings_not_canceled': 690000,
    'agent': 600,
    'company': 600,
    'adr': 600,
    'required_car_parking_spaces': 3000,
    'total_of_special_requests': 5000
}

# Descriptions for each column
column_descriptions = {
    'hotel': 'Resort Hotel, City Hotel',
    'meal': 'Type of meal booked - Bed & Breakfast,Full Board(breakfast, lunch & dinner), Half Board(breakfast & dinner), Undefined/SC(no meal package)',
    'market_segment': '0: Direct, Corporate, Online TA(Travel Agents), Offline TA/TO(Tour Operators), Complementary, Groups, Undefined, Aviation',
    'distribution_channel': 'Direct, Corporate, TA(Travel Agents)/TO(Tour Operators), Undefined, GDS',
    'reserved_room_type': 'C, A, D, E, G, F, H, L, B',
    'deposit_type': 'No Deposit, Refundable, Non Refund',
    'customer_type': 'Transient, Contract, Transient-Party, Group',
    'year': 'The year in which the reservation status was updated',
    'lead_time': 'Number of days that elapsed between the entering date of the booking into the PMS (Property Management System) and the arrival date.',
    'is_repeated_guest': 'Whether the guest is a repeated guest or not.',
    'agent': 'ID of the travel agent responsible for the booking',
    'adults': 'Number of adults included in the booking.',
    'company': 'ID of the company responsible for the booking',
    'adr': 'Average Daily Rates described by dividing the sum of all accommodations transactions by the total number of staying nights.',
    'total_of_special_requests': 'Total unique requests from consumers(e.g. high floor, view from the room, etc)',
    'day': 'The day of the month in which the reservation status was updated',
    'month': 'The month in which the reservation status was updated ',
    'stays_in_weekend_nights': 'Number of weekend nights the guest stayed or booked to stay at the hotel.',
    'stays_in_week_nights': 'Number of week nights the guest stayed or booked to stay at the hotel.',
    'arrival_date_week_number': 'The week number of the arrival date.',
    'children': 'Number of children included in the booking.',
    'babies': 'Number of babies included in the booking.',
    'previous_cancellations': 'Number of previous bookings that were cancelled by the customer prior to the current booking.',
    'previous_bookings_not_canceled': 'Number of previous bookings that were not cancelled by the customer prior to the current booking.',
    'required_car_parking_spaces': 'Number of car parking spaces required by the customer.',
    'arrival_date_day_of_month': 'The day of the month of the arrival date.'
}


################
def fetch_data():
    try:
        data = list(collection.find())
        if not data:
            st.error("No data found in the collection.")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Clean and denormalize specific columns
        for col in ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'adr', 'company', 'agent']:
            if col in df.columns:
                df[col] = df[col].apply(denormalize_value)
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, setting errors to NaN

        return df
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()


def get_week_number(year, month, day):
    import datetime

    # Get the date object
    date_obj = datetime.date(year, month, day)

    # Calculate the week number using ISO week date format
    week_number = date_obj.isocalendar()[1]

    return week_number


def get_week_options():
    import datetime
    current_year = datetime.datetime.now().year
    return [calendar.month_name[month] for month in range(1, 13)]


def get_day_options(month):
    import datetime
    current_year = datetime.datetime.now().year
    last_day = calendar.monthrange(current_year, month)[1]
    return list(range(1, last_day + 1))


def get_lead_time_label(year_diff):
    if year_diff == 0:
        return "Lead Time for This Year"
    elif year_diff == 1:
        return "Lead Time for Next Year"
    else:
        return f"Lead Time for {year_diff} Years Later"


def get_lead_time(year, month, day, arrival_month, arrival_day, num_years=5):
    try:
        reservation_date = datetime(year, month, day)
    except ValueError:
        raise ValueError("Invalid date. Please enter a valid date.")

    current_date = datetime.now()
    current_year = current_date.year
    suggested_lead_times = []

    for i in range(num_years):
        arrival_year = current_year + i
        year_diff = arrival_year - current_year

        first_day_of_year = datetime(arrival_year, 1, 1)
        days_to_month = (arrival_month - 1) * 30  # Assuming each month has 30 days for simplicity
        first_day_of_month = first_day_of_year + timedelta(days=days_to_month)
        arrival_date = first_day_of_month + timedelta(days=arrival_day - 1)

        lead_time = (arrival_date - reservation_date).days

        if arrival_year == year and reservation_date <= arrival_date:
            lead_time = max(lead_time, 0)

        if lead_time >= 0:
            lead_time_label = get_lead_time_label(year_diff)
            suggested_lead_times.append((lead_time_label, lead_time))

    return suggested_lead_times


def normalize_value(value):
    return np.log(value + 1)


# Function to denormalize values
def denormalize_value(value):
    try:
        if value is not None and isinstance(value, (int, float)):
            if np.isfinite(value):
                return round(np.exp(value) - 1, 2)  # Rounding to 2 decimal places
            else:
                return "Infinity" if value > 0 else "-Infinity"  # Handle infinity values
        else:
            return value
    except Exception as e:
        st.error(f"An error occurred while denormalizing the value: {e}")
        return value


def internal_to_user_year(internal_year):
    """Convert internal year format to user-friendly year format."""
    return internal_year + 2014


def user_to_internal_year(user_year):
    """Convert user-friendly year format to internal year format."""
    return user_year - 2014


def insert_customer_data(customer_id, customer_data):
    collection.insert_one({"_id": ObjectId(customer_id), **customer_data})
    st.success("Customer data inserted successfully! ID: {}".format(customer_id))


def reverse_mapping(data, mapping):
    return mapping.get(data, data)


#################
# Define function to make predictions
def predict_cancelation(data):
    prediction = model.predict_proba(data)
    return prediction


# Function to check for changes in the database
def check_changes(client_ip):
    document_before = collection.find_one({"_id": client_ip})
    document_after = None  # Fetch the latest document from the database
    if document_before != document_after:
        return document_before, document_after
    else:
        return None, None


# Get client IP address
def get_client_ip():
    return st.experimental_get_query_params().get("client_ip", [None])[0]


# Function to copy the selected customer ID to clipboard
def copy_customer_id(customer_id):
    pyperclip.copy(str(customer_id))
    st.success("Customer ID copied to clipboard!")


def get_max_values():
    max_values = {}
    numerical_features = {
        'lead_time': float,
        'adr': float,
        'total_of_special_requests': float,
        'stays_in_weekend_nights': int,
        'stays_in_week_nights': int,
        'arrival_date_week_number': float,
        'arrival_date_day_of_month': float,
        'adults': int,
        'children': float,
        'babies': int,
        'agent': float,
        'company': float,
        'adr': float,
        'total_of_special_requests': int,
        'required_car_parking_spaces': int,
        'previous_cancellations': int,
        'previous_bookings_not_canceled': int
    }

    for feature, value_type in numerical_features.items():
        max_value = collection.find_one(sort=[(feature, pymongo.DESCENDING)])[feature]
        if value_type == int:
            max_values[feature] = int(max_value)
        else:
            max_values[feature] = float(max_value)
    return max_values


def get_total_customers():
    return collection.count_documents({})


def extract_customer_id(selected_customer):
    match = re.search(r'\(([^)]+)\)', selected_customer)
    if match:
        return match.group(1)
    return None


# Function to load data from MongoDB with optional filters
def load_data(filters=None):
    if filters:
        data = list(collection.find(filters))
    else:
        data = list(collection.find())
    return pd.DataFrame(data)


def database_screen(customer_object_id):
    # Define gradient background colors
    gradient_color1 = '#4b6cb7'  # Deep blue
    gradient_color2 = '#182848'  # Dark blue

    # Apply gradient background using custom CSS
    st.markdown(
        f"""
        <style>
            .database-container {{
                background: linear-gradient(45deg, {gradient_color1}, {gradient_color2}) no-repeat;
                background-size: cover;
                padding: 2rem;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }}
            .group-container {{
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    header_html = """
        <div style="display: flex; justify-content: center;">
            <h1 style="text-align: center;">Customer Database</h1>
        </div>
    """

    # Display the centered header
    st.write(header_html, unsafe_allow_html=True)

    # Dictionary to map feature names to descriptions, values, and their corresponding meanings
    feature_info = {
        'hotel': {'description': 'Hotel',
                  'info': 'Type of hotel where the booking was made. Resort Hotel refers to a hotel located in a resort area, while City Hotel refers to a hotel located in an urban area.',
                  'values': {'0': 'Resort Hotel', '1': 'City Hotel'}},
        'meal': {'description': 'Meal',
                 'info': 'Type of meal booked for the stay. Options include Bed & Breakfast, Full Board (breakfast, lunch, and dinner), Half Board (breakfast and one other meal, usually dinner), and Undefined/SC (no meal package specified).',
                 'values': {'0': 'Bed & Breakfast', '1': 'Full Board', '2': 'Half Board', '3': 'Undefined/SC'}},
        'market_segment': {'description': 'Market Segment',
                           'info': 'Segmentation of the market based on customer types. Options include Direct (booking made directly by the customer), Corporate (Corporate account), Online TA (Travel agency), Offline TA/TO (Traditional travel agency or Tour operator), Complementary (Complementary arrangement), Groups (booking made for a group of people), Undefined (undefined market segment), and Aviation (Aviation company).',
                           'values': {'0': 'Direct', '1': 'Corporate', '2': 'Online TA', '3': 'Offline TA/TO',
                                      '4': 'Complementary', '5': 'Groups', '6': 'Undefined', '7': 'Aviation'}},
        'distribution_channel': {'description': 'Distribution Channel',
                                 'info': 'Channel through which the booking was distributed. Options include Direct (booking made directly through the hotel), Corporate (booking made through a corporate account), TA/TO (booking made through a travel agency or tour operator), Undefined (undefined distribution channel), and GDS (booking made through a Global Distribution System).',
                                 'values': {'0': 'Direct', '1': 'Corporate', '2': 'TA/TO', '3': 'Undefined',
                                            '4': 'GDS'}},
        'reserved_room_type': {'description': 'Reserved Room Type', 'info': 'Type of room reserved for the stay.',
                               'values': {'0': 'C', '1': 'A', '2': 'D', '3': 'E', '4': 'G', '5': 'F', '6': 'H',
                                          '7': 'L', '8': 'B'}},
        'deposit_type': {'description': 'Deposit Type',
                         'info': 'Type of deposit made for the booking. Options include No Deposit, Refundable, and Non-Refundable.',
                         'values': {'0': 'No Deposit', '1': 'Refundable', '2': 'Non Refund'}},
        'customer_type': {'description': 'Customer Type',
                          'info': 'Type of customer making the booking. Options include Transient (individual booking), Contract (booking made under a contract), Transient-Party (group booking with individual pricing and policies), and Group (booking made for a group).',
                          'values': {'0': 'Transient', '1': 'Contract', '2': 'Transient-Party', '3': 'Group'}},
        'year': {'description': 'Year', 'info': 'Year in which the reservation status was updated.',
                 'values': {'0': '2014', '1': '2015', '2': '2016', '3': '2017', '4': '2018', '5': '2019', '6': '2020',
                            '7': '2021', '8': '2022', '9': '2023', '10': '2024', '11': '2025', '12': '2026',
                            '13': '2027', '14': '2028', '15': '2029', '16': '2030', '17': '2031', '18': '2032',
                            '19': '2033', '20': '2034', '21': '2035', '22': '2036', '23': '2037', '24': '2038',
                            '25': '2039', '26': '2040', '27': '2041', '28': '2042', '29': '2043', '30': '2044'}},
        'month': {'description': 'Month', 'info': 'Month in which the reservation status was updated.', 'values': {}},
        # Add values if necessary
        'day': {'description': 'Day', 'info': 'Day of the month in which the reservation status was updated.',
                'values': {}},  # Add values if necessary
        'lead_time': {'description': 'Lead Time',
                      'info': 'Number of days between the booking date and the arrival date.', 'values': {}},
        # Add values if necessary
        'stays_in_weekend_nights': {'description': 'Stays in Weekend Nights',
                                    'info': 'Number of weekend nights the guest stayed or booked to stay at the hotel.',
                                    'values': {}},  # Numeric representation of weekend nights stayed
        'stays_in_week_nights': {'description': 'Stays in Week Nights',
                                 'info': 'Number of week nights the guest stayed or booked to stay at the hotel.',
                                 'values': {}},  # Numeric representation of week nights stayed
        'arrival_date_week_number': {'description': 'Arrival Date Week Number',
                                     'info': 'Week number of the arrival date.', 'values': {}},
        # Numeric representation of the week number
        'arrival_date_day_of_month': {'description': 'Arrival Date Day of Month',
                                      'info': 'Day of the month of the arrival date.', 'values': {}},
        # Numeric representation of the day of the month
        'adults': {'description': 'Adults', 'info': 'Number of adults included in the booking.', 'values': {}},
        # Numeric representation of the number of adults
        'children': {'description': 'Children', 'info': 'Number of children included in the booking.', 'values': {}},
        # Numeric representation of the number of children
        'babies': {'description': 'Babies', 'info': 'Number of babies included in the booking.', 'values': {}},
        # Numeric representation of the number of babies
        'is_repeated_guest': {'description': 'Is Repeated Guest',
                              'info': 'Indicator if the guest is a repeated guest (1) or not (0).',
                              'values': {'0': 'Not a Repeated Guest', '1': 'Repeated Guest'}},
        'agent': {'description': 'Agent', 'info': 'ID of the travel agent responsible for the booking.', 'values': {}},
        # Numeric representation of the agent ID
        'company': {'description': 'Company', 'info': 'ID of the company responsible for the booking.', 'values': {}},
        # Numeric representation of the company ID
        'adr': {'description': 'Price per Day (ADR)',
                'info': 'Average Daily Rate (ADR) calculated by dividing the sum of all accommodation transactions by the total number of staying nights.',
                'values': {}},  # Numeric representation of the average daily rate
        'total_of_special_requests': {'description': 'Total of Special Requests',
                                      'info': 'Total number of unique requests from customers, such as high floor, room view, etc.',
                                      'values': {}},  # Numeric representation of the total number of special requests
        'required_car_parking_spaces': {'description': 'Required Car Parking Spaces',
                                        'info': 'Number of car parking spaces requested by the customer.',
                                        'values': {}},  # Numeric representation of the number of car parking spaces
        'previous_cancellations': {'description': 'Previous Cancellations',
                                   'info': 'Number of previous bookings that were cancelled by the customer prior to the current booking.',
                                   'values': {}},  # Numeric representation of the number of previous cancellations
        'previous_bookings_not_canceled': {'description': 'Previous Bookings Not Canceled',
                                           'info': 'Number of previous bookings that were not cancelled by the customer prior to the current booking.',
                                           'values': {}},
    }

    # Fetch all customer IDs from the database
    # Fetch all customers from the database
    customers = list(collection.find())

    # Create customer list with names and IDs
    customer_list = ["None"] + [
        f"{customer.get('name', 'Unknown')} {customer.get('surname', 'Unknown')} ({str(customer['_id'])})" for customer
        in customers]

    # Selection box to choose a customer
    selected_customer = st.selectbox('Select Customer', customer_list, key='select_customer')

    # If a customer is selected and it's not 'None', display their information
    if selected_customer and selected_customer != 'None':
        # Extract the customer ID from the selected item
        selected_customer_id = selected_customer.split('(')[-1].strip(')')

        # Fetch customer data from the database
        customer_data = collection.find_one({"_id": ObjectId(selected_customer_id)})
        if customer_data:
            # Denormalize specific fields
            fields_to_denormalize = ['lead_time', 'children', 'arrival_date_week_number', 'arrival_date_day_of_month',
                                     'agent', 'adr', 'company']
            for field in fields_to_denormalize:
                if field in customer_data:
                    customer_data[field] = int(round(denormalize_value(customer_data[field])))
            # Define colors for each group
            group_colors = ['#6dcff6', '#6dcff6', '#6dcff6', '#6dcff6', '#6dcff6']

            group1_features = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                               'deposit_type', 'customer_type']
            group2_features = ['year', 'month', 'day', 'lead_time']
            group3_features = ['adults', 'children', 'babies']
            group4_features = ['stays_in_weekend_nights', 'stays_in_week_nights', 'arrival_date_week_number',
                               'arrival_date_day_of_month']
            group5_features = ['is_repeated_guest', 'agent', 'company', 'adr', 'total_of_special_requests',
                               'required_car_parking_spaces', 'previous_cancellations',
                               'previous_bookings_not_canceled']

            groups = [group1_features, group2_features, group3_features, group4_features, group5_features]
            group_titles = [
                "Booking Information",
                "Date-related Fields",
                "Guest Details",
                "Stay Information",
                "Additional Details"
            ]

            for group_idx, (group_features, group_title) in enumerate(zip(groups, group_titles)):
                group_color = group_colors[group_idx]

                st.markdown(
                    f"<div class='database-container' style='border: 2px solid {group_color};'>"
                    f"<h3 style='color: white; text-align: center;'>{group_title}</h3>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                for feature in group_features:
                    if feature in feature_info:
                        feature_description = feature_info[feature]['description']
                        value = customer_data.get(feature, '')
                        value_description = feature_info[feature].get('values', {}).get(str(value), value)
                        if isinstance(value, str):
                            value_str = f"{value_description} ({value})"
                        elif isinstance(value, int) and '.' not in str(value):
                            value_str = f"{value_description} ({value})" if value_description != value else value_description
                        else:
                            value_str = value_description
                        st.write(
                            f"<div style='text-align: center;'><b>{feature_description}:</b> <span style='color: {group_color}; font-weight: bold;'>{value_str}</span></div>",
                            unsafe_allow_html=True)
                        if 'info' in feature_info[feature]:
                            st.write(
                                f"<div style='text-align: center; margin-top: 5px;'>{feature_info[feature]['info']}</div>",
                                unsafe_allow_html=True)
                        st.markdown('---', unsafe_allow_html=True)  # Add horizontal line as a visual separator

            # Initialize session state
            if 'enter_pressed' not in st.session_state:
                st.session_state.enter_pressed = False

            # Extract the customer ID
            customer_id = extract_customer_id(selected_customer)

            # Button to copy and apply customer ID
            if customer_id:
                st.write(f"Selected Customer: {selected_customer}")
                copy_button_key = f'copy_and_paste_customer_id_{customer_id}'  # Unique key for each button
                if st.button('Apply Information of This Customer', key=copy_button_key):
                    # Copy the ID to clipboard
                    pyperclip.copy(customer_id)
                    # Store the copied ID in session state
                    st.session_state.customer_object_id = customer_id
                    st.success('Customer ID applied and copied to clipboard!')
            else:
                st.error("Customer ID could not be extracted. Please check the selected customer string.")

            # Input box to enter customer's ObjectID
            if 'customer_object_id' not in st.session_state:
                st.session_state['customer_object_id'] = ""

            customer_object_id = st.text_input('Enter Customer ObjectID', value=st.session_state['customer_object_id'])

            # Check if Enter key is pressed
            if st.session_state.enter_pressed:
                # Handle the action to open the new page
                # You can place your code here to open the new page or perform any other action
                st.write("Enter key pressed!")

            # Inject JavaScript to detect Enter key press
            st.markdown(
                """
                <script>
                    document.addEventListener('keydown', function(event) {
                        if (event.key === "Enter") {
                            // Set the enter_pressed state to true
                            Shiny.setInputValue('enter_pressed', true);
                        }
                    });
                </script>
                """,
                unsafe_allow_html=True
            )

            if customer_object_id:
                try:
                    # Attempt to fetch customer information based on the client IP
                    customer_data = collection.find_one({"_id": ObjectId(customer_object_id)})

                    if customer_data:
                        # Remove the ObjectID from customer_data
                        customer_data.pop('_id', None)

                    else:
                        st.error(
                            'No customer found with the entered ObjectID. Please enter a valid customer ObjectID.')
                except Exception as e:
                    st.error(f'Error: {e}. Please enter a valid ObjectID.')

            else:
                st.write('Please enter a valid customer ObjectID.')

                # Stop further execution if no ID is provided
                return

            # Define tabs
            tabs = ["Booking Information", "Date-related Fields", "Guest Details", "Stay Information",
                    "Additional Details"]
            selected_tab = st.selectbox("Select Page", tabs)

            # Initialize year outside if-elif blocks
            year = 0
            month = 0
            day = 0
            hotel = 0
            lead_time = 0
            meal = 0
            market_segment = 0
            distribution_channel = 0
            reserved_room_type = 0
            selected_deposit_type = 0
            customer_type = 0
            arrival_date_week_number = 0
            arrival_date_day_of_month = 0
            stays_in_weekend_nights = 0
            stays_in_week_nights = 0
            adults = 0
            children = 0
            is_repeated_guest = 0
            previous_cancellations = 0
            previous_bookings_not_canceled = 0
            agent = 0
            company = 0
            adr = 0
            required_car_parking_spaces = 0
            total_of_special_requests = 0
            babies = 0

            # Booking Information
            if selected_tab == "Booking Information":
                header_html = """
                            <div style="display: flex; justify-content: center;">
                                <h1 style="text-align: center;">Booking Information</h1>
                            </div>
                        """

                # Display the centered header
                st.write(header_html, unsafe_allow_html=True)
                st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                with col1:
                    if customer_data.get('hotel', 0) == 0:
                        st.image("resort.png", width=75)  # Display a resort icon if hotel is 0 (Resort Hotel)
                    else:
                        st.image("city.png", width=75)  # Display a city icon if hotel is 1 (City Hotel)

                    hotel = st.selectbox('Hotel', options=[0, 1],
                                         format_func=lambda x: 'Resort Hotel' if x == 0 else 'City Hotel',
                                         help=column_descriptions['hotel'], index=customer_data.get('hotel', 0))

                    st.image("food.png", width=75)
                    meal = st.selectbox('Meal', options=[0, 1, 2, 3],
                                        format_func=lambda x:
                                        ['Bed & Breakfast', 'Full Board', 'Half Board', 'Undefined/SC'][
                                            x],
                                        help=column_descriptions['meal'], index=customer_data.get('meal', 0))

                with col2:
                    st.image("briefcase.png", width=75)
                    market_segment = st.selectbox('Market Segment', options=[0, 1, 2, 3, 4, 5, 6, 7],
                                                  format_func=lambda x:
                                                  ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO',
                                                   'Complementary', 'Groups', 'Undefined', 'Aviation'][x],
                                                  help=column_descriptions['market_segment'],
                                                  index=customer_data.get('market_segment', 0))

                    if customer_data.get('distribution_channel', 0) == 0:
                        st.image("travel_agency.png",
                                 width=75)  # Display a travel agency icon if distribution channel is 0
                    else:
                        st.image("computer.png", width=75)
                    distribution_channel = st.selectbox('Distribution Channel', options=[0, 1, 2, 3, 4],
                                                        format_func=lambda x:
                                                        ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'][x],
                                                        help=column_descriptions['distribution_channel'],
                                                        index=customer_data.get('distribution_channel', 0))

                with col3:
                    st.image("roomtype.png", width=75)
                    reserved_room_type = st.selectbox('Reserved Room Type', options=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                      format_func=lambda x:
                                                      ['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'B'][x],
                                                      help=column_descriptions['reserved_room_type'],
                                                      index=customer_data.get('reserved_room_type', 0))

                    # Deposit Type Icons
                    deposit_type_options = ['No Deposit', 'Refundable', 'Non Refund']
                    deposit_type_value = customer_data.get('deposit_type', 0)
                    if deposit_type_value == 0:
                        st.image("nodeposit.png", width=75)  # Display a no deposit icon if deposit type is 0
                        selected_deposit_type = 'No Deposit'
                    elif deposit_type_value == 1:
                        st.image("refundable.png",
                                 width=75)  # Display a refundable deposit icon if deposit type is 1
                        selected_deposit_type = 'Refundable'
                    elif deposit_type_value == 3:
                        st.image("nonrefund.png",
                                 width=75)  # Display a non-refundable deposit icon if deposit type is 3
                        selected_deposit_type = 'Non Refund'

                    selected_deposit_type = st.selectbox('Deposit Type', options=deposit_type_options,
                                                         help=column_descriptions['deposit_type'],
                                                         index=deposit_type_options.index(selected_deposit_type))

                with col4:
                    customer_type_options = ['Transient', 'Contract', 'Transient-Party', 'Group']
                    customer_type_value = customer_data.get('customer_type', 0)
                    if customer_type_value == 0:
                        st.image("transient.png", width=75)  # Display a transient icon if customer type is 0
                        selected_customer_type = 'Transient'
                    elif customer_type_value == 1:
                        st.image("contract.png", width=75)  # Display a contract icon if customer type is 1
                        selected_customer_type = 'Contract'
                    elif customer_type_value in [2, 3]:
                        st.image("groupofpeople.png", width=75)  # Display a group icon if customer type is 2 or 3
                        selected_customer_type = 'Group'

                    customer_type = st.selectbox('Customer Type', options=[0, 1, 2, 3],
                                                 format_func=lambda x:
                                                 ['Transient', 'Contract', 'Transient-Party', 'Group'][x],
                                                 help=column_descriptions['customer_type'],
                                                 index=customer_data.get('customer_type', 0))
                st.write('---')


            elif selected_tab == "Date-related Fields":
                header_html = """
                                            <div style="display: flex; justify-content: center;">
                                                <h1 style="text-align: center;">Date-related Fields</h1>
                                            </div>
                                        """

                # Display the centered header
                st.write(header_html, unsafe_allow_html=True)
                st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

                # Create a layout with photo on the left and specifications on the right
                col1, col2 = st.columns([2, 3])

                # Display calendar image on the left
                with col1:
                    st.image("calendar.png", width=230)

                # Create a container for the specifications on the right
                with col2:
                    # Specifications stacked vertically
                    st.subheader("Date Specifications")

                    # Input for Month
                    month = st.number_input('Month', min_value=0, max_value=max_values['month'],
                                            help=column_descriptions['month'], value=customer_data.get('month', 0))

                    # Input for Day
                    day = st.number_input('Day', min_value=0, max_value=max_values['day'],
                                          help=column_descriptions['day'],
                                          value=customer_data.get('day', 0))

                    # Input for Year
                    year = st.selectbox('Year',
                                        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                        format_func=lambda x:
                                        ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023',
                                         '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033',
                                         '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043',
                                         '2044'][x],
                                        help=column_descriptions['year'], index=customer_data.get('year', 0))

                    # Input for Lead Time
                    # Input for Lead Time
                    lead_time = st.number_input('Lead Time', min_value=0.0,
                                                max_value=float(max_values['lead_time']), step=0.01,
                                                help=column_descriptions['lead_time'],
                                                value=denormalize_value(customer_data.get('lead_time', 0)))

                st.write('---')

            elif selected_tab == "Guest Details":
                header_html = """
                                                            <div style="display: flex; justify-content: center;">
                                                                <h1 style="text-align: center;">Guest Details</h1>
                                                            </div>
                                                        """

                # Display the centered header
                st.write(header_html, unsafe_allow_html=True)
                st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

                # Create a layout with images and input fields
                col1, col2, col3 = st.columns([1, 1, 1])

                # Display image for Adults
                with col1:
                    st.image("adults.png", width=75)

                # Input field for Adults
                with col1:
                    adults = st.number_input('Adults',
                                             min_value=0,  # Keep min_value as an integer
                                             max_value=int(max_values['adults']),  # Convert max_value to integer
                                             help=column_descriptions['adults'],
                                             value=int(customer_data.get('adults', 0)))  # Convert value to integer

                # Display image for Children
                with col2:
                    st.image("children.png", width=75)

                # Input field for Children
                with col2:
                    children = st.number_input(
                        'Children',
                        min_value=0,
                        max_value=int(round(max_values['children'])),
                        help=column_descriptions['children'],
                        value=int(round(denormalize_value(customer_data.get('children', 0))))
                    )

                # Display image for Babies
                with col3:
                    st.image("baby.png", width=75)

                # Input field for Babies
                with col3:
                    babies = st.number_input('Babies',
                                             min_value=0,  # Keep min_value as an integer
                                             max_value=int(max_values['babies']),  # Convert max_value to integer
                                             help=column_descriptions['babies'],
                                             value=int(customer_data.get('babies', 0)))  # Convert value to integer

                st.write('---')

            elif selected_tab == "Stay Information":
                header_html = """
                                                                            <div style="display: flex; justify-content: center;">
                                                                                <h1 style="text-align: center;">Stay Information</h1>
                                                                            </div>
                                                                        """

                # Display the centered header
                st.write(header_html, unsafe_allow_html=True)
                st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

                # Create a layout with images and input fields
                col1, col2 = st.columns([1, 1])

                # Display image for Stays in Weekend Nights and Stays in Week Nights
                with col1:
                    st.image("night.png", width=100)

                # Input fields for Stays in Weekend Nights and Stays in Week Nights
                with col1:
                    stays_in_weekend_nights = st.number_input('Stays in Weekend Nights', min_value=0.0,
                                                              max_value=float(max_values['stays_in_weekend_nights']),
                                                              help=column_descriptions['stays_in_weekend_nights'],
                                                              value=float(
                                                                  customer_data.get('stays_in_weekend_nights', 0)))

                    stays_in_week_nights = st.number_input('Stays in Week Nights', min_value=0.0,
                                                           # Make min_value a float
                                                           max_value=float(max_values['stays_in_week_nights']),
                                                           help=column_descriptions['stays_in_week_nights'],
                                                           value=float(customer_data.get('stays_in_week_nights', 0)))

                # Display image for Arrival Date Week Number and Arrival Date Day of Month
                with col2:
                    st.image("arrival.png", width=100)

                # Input fields for Arrival Date Week Number and Arrival Date Day of Month
                with col2:
                    arrival_date_week_number = st.number_input('Arrival Date Week Number', min_value=0.0,
                                                               max_value=float(max_values['arrival_date_week_number']),
                                                               step=0.01,
                                                               value=denormalize_value(
                                                                   min(customer_data.get('arrival_date_week_number', 0),
                                                                       float(max_values['arrival_date_week_number']))),
                                                               help=column_descriptions['arrival_date_week_number'])

                    arrival_date_day_of_month = st.number_input('Arrival Date Day of Month', min_value=0.0,
                                                                max_value=float(
                                                                    max_values['arrival_date_day_of_month']),
                                                                step=0.01,
                                                                value=denormalize_value(
                                                                    min(customer_data.get('arrival_date_day_of_month',
                                                                                          0),
                                                                        float(
                                                                            max_values['arrival_date_day_of_month']))),
                                                                help=column_descriptions['arrival_date_day_of_month'])

                st.write('---')

            elif selected_tab == "Additional Details":
                header_html = """
                                                                                            <div style="display: flex; justify-content: center;">
                                                                                                <h1 style="text-align: center;">Additional Details</h1>
                                                                                            </div>
                                                                                        """

                # Display the centered header
                st.write(header_html, unsafe_allow_html=True)
                st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

                # Create a layout with images and input fields
                col1, col2, col3 = st.columns([1, 1, 1])

                # Display image for Is Repeated Guest
                with col1:
                    st.image("yes.png" if customer_data.get('is_repeated_guest', 0) == 1 else "no.png", width=75)

                # Input field for Is Repeated Guest
                with col1:
                    is_repeated_guest = st.selectbox('Is Repeated Guest', options=[0, 1],
                                                     format_func=lambda x: 'No' if x == 0 else 'Yes',
                                                     help=column_descriptions['is_repeated_guest'],
                                                     index=customer_data.get('is_repeated_guest', 0))

                # Display image for Agent
                with col2:
                    st.image("agent.png", width=75)

                # Input field for Agent
                with col2:
                    agent = st.number_input('Agent', min_value=0.0, max_value=float(max_values['agent']), step=0.01,
                                            value=denormalize_value(
                                                min(customer_data.get('agent', 0), float(max_values['agent']))),
                                            help=column_descriptions['agent'])

                # Display image for Company
                with col3:
                    st.image("company.png", width=75)

                # Input field for Company
                with col3:
                    company = st.number_input('Company', min_value=0.0, max_value=float(max_values['company']),
                                              step=0.01,
                                              value=denormalize_value(
                                                  min(customer_data.get('company', 0), float(max_values['company']))),
                                              help=column_descriptions['company'])

                # Create a layout with images and input fields
                col1, col2, col3 = st.columns([1, 1, 1])

                # Display image for ADR
                with col1:
                    st.image("price1.png", width=75)

                # Input field for ADR
                with col1:
                    adr = st.number_input('Price per Day (ADR)', min_value=0.0, max_value=float(max_values['adr']),
                                          step=0.01,
                                          value=denormalize_value(
                                              min(customer_data.get('adr', 0), float(max_values['adr']))),
                                          help=column_descriptions['adr'])

                # Display image for Total of Special Requests
                with col2:
                    st.image("special.png", width=75)

                # Input field for Total of Special Requests
                with col2:
                    total_of_special_requests = st.number_input('Total of Special Requests',
                                                                min_value=0,  # Keep min_value as an integer
                                                                max_value=max_values['total_of_special_requests'],
                                                                help=column_descriptions['total_of_special_requests'],
                                                                value=int(customer_data.get('total_of_special_requests',
                                                                                            0)))  # Convert value to integer

                # Display image for Required Car Parking Spaces
                with col3:
                    st.image("parking.png", width=75)

                # Input field for Required Car Parking Spaces
                with col3:
                    required_car_parking_spaces = st.number_input('Required Car Parking Spaces',
                                                                  min_value=0,  # Keep min_value as an integer
                                                                  max_value=max_values['required_car_parking_spaces'],
                                                                  help=column_descriptions[
                                                                      'required_car_parking_spaces'],
                                                                  value=int(
                                                                      customer_data.get('required_car_parking_spaces',
                                                                                        0)))  # Convert value to integer

                # Create a layout with input fields for Previous Cancellations and Previous Bookings Not Canceled
                col1, col2 = st.columns(2)

                # Input field for Previous Cancellations
                with col1:
                    previous_cancellations = st.number_input('Previous Cancellations',
                                                             min_value=0,  # Keep min_value as an integer
                                                             max_value=max_values['previous_cancellations'],
                                                             help=column_descriptions['previous_cancellations'],
                                                             value=int(customer_data.get('previous_cancellations',
                                                                                         0)))  # Convert value to integer

                # Input field for Previous Bookings Not Canceled
                with col2:
                    previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled',
                                                                     min_value=0,  # Keep min_value as an integer
                                                                     max_value=max_values[
                                                                         'previous_bookings_not_canceled'],
                                                                     help=column_descriptions[
                                                                         'previous_bookings_not_canceled'],
                                                                     value=int(customer_data.get(
                                                                         'previous_bookings_not_canceled',
                                                                         0)))  # Convert value to integer

            if st.button('Predict Cancellation'):
                # Convert input data to DataFrame
                input_data = pd.DataFrame({
                    'hotel': [hotel],
                    'meal': [meal],
                    'market_segment': [market_segment],
                    'distribution_channel': [distribution_channel],
                    'reserved_room_type': [reserved_room_type],
                    'deposit_type': [selected_deposit_type],
                    'customer_type': [customer_type],
                    'year': [year],
                    'month': [month],
                    'day': [day],
                    'lead_time': [lead_time],
                    'arrival_date_week_number': [arrival_date_week_number],
                    'arrival_date_day_of_month': [arrival_date_day_of_month],
                    'stays_in_weekend_nights': [stays_in_weekend_nights],
                    'stays_in_week_nights': [stays_in_week_nights],
                    'adults': [adults],
                    'children': [children],
                    'babies': [babies],
                    'is_repeated_guest': [is_repeated_guest],
                    'previous_cancellations': [previous_cancellations],
                    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
                    'agent': [agent],
                    'company': [company],
                    'adr': [adr],
                    'required_car_parking_spaces': [required_car_parking_spaces],
                    'total_of_special_requests': [total_of_special_requests]
                })

                # Before the prediction, ensure only numeric columns are passed to the model
                # Define the list of columns to exclude
                exclude_columns = ['name', 'surname', '_id']

                # Remove excluded columns from customer_data
                customer_data_filtered = {k: v for k, v in customer_data.items() if k not in exclude_columns}

                # Convert the filtered customer data to a DataFrame
                input_data = pd.DataFrame([customer_data_filtered])

                # Make prediction
                prediction = model.predict_proba(input_data)
                cancellation_probability = prediction[0][1] * 100  # Probability of cancellation
                st.write(f'Cancellation Probability: {cancellation_probability:.2f}%')



        else:
            st.error('No customer found with the selected ObjectID.')
    elif selected_customer == 'None':
        st.warning("Please select a customer to view their information.")


def prediction_random_values_screen():
    header_html = """
                <div style="display: flex; justify-content: center;">
                    <h1 style="text-align: center;">Hotel Booking Cancellation Prediction Simulation</h1>
                </div>
            """

    # Display the centered header
    st.write(header_html, unsafe_allow_html=True)

    # Input form for random values
    col1, col2 = st.columns(2)

    with st.form(key='input_form'):
        with col1:
            st.markdown("---")
            hotel = st.selectbox('Hotel', options=['Resort Hotel', 'City Hotel'])
            meal = st.selectbox('Meal', options=['Bed & Breakfast', 'Full Board', 'Half Board', 'Undefined/SC'])
            market_segment = st.selectbox('Market Segment',
                                          options=['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary',
                                                   'Groups', 'Undefined', 'Aviation'])
            distribution_channel = st.selectbox('Distribution Channel',
                                                options=['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'])
            reserved_room_type = st.selectbox('Reserved Room Type',
                                              options=['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'B'])
            deposit_type = st.selectbox('Deposit Type', options=['No Deposit', 'Refundable', 'Non Refund'])
            customer_type = st.selectbox('Customer Type', options=['Transient', 'Contract', 'Transient-Party', 'Group'])
            year = st.text_input('Year (The Year the Reservation was Made or Updated)')
            month = st.slider('Month (The Month the Reservation was Made or Updated)', 1, 12)
            day = st.slider('Day (The Day of the Month the Reservation was Made or Updated)', 1, 31)

            try:
                year = int(year) - 2014
            except ValueError:
                year = 0  # Default to 0 if the input is invalid

            month_options = get_week_options()
            selected_month = st.selectbox('Select Arrival Month', options=month_options)
            month_number = list(calendar.month_name).index(selected_month)

            day_options = get_day_options(month_number)
            selected_day = st.selectbox('Select Arrival Date Day', options=day_options)

            lead_time = st.text_input('Lead Time', value='')

        with col2:
            st.markdown("---")
            stays_in_weekend_nights = st.text_input('Stays in Weekend Nights')
            stays_in_week_nights = st.text_input('Stays in Week Nights')
            adults = st.text_input('Number of Adults')
            children = st.text_input('Number of Children')
            babies = st.text_input('Number of Babies')
            is_repeated_guest = st.radio('Is Repeated Guest', options=['No', 'Yes'])
            previous_cancellations = st.text_input('Previous Cancellations')
            previous_bookings_not_canceled = st.text_input('Previous Bookings Not Canceled')
            agent = st.text_input('Agent')
            company = st.text_input('Company')
            adr = st.text_input('ADR')
            required_car_parking_spaces = st.text_input('Required Car Parking Spaces')
            total_of_special_requests = st.text_input('Total of Special Requests')

        submit_button = st.form_submit_button("Predict With These Features")

    get_lead_time_button = st.button('Get Lead Time Suggestions', key="get_lead_time_button")
    if get_lead_time_button:
        if all([year, month, day, selected_month, selected_day]):
            try:
                year = int(year)
                month = int(month)
                day = int(day)
                selected_month = month_number
                selected_day = int(selected_day)

                # Store suggested lead times
                suggested_lead_times = get_lead_time(year, month, day, selected_month, selected_day)
                if suggested_lead_times:  # Check if there are suggested lead times
                    for i, (lead_time_label, lead_time_value) in enumerate(suggested_lead_times, start=1):
                        corrected_lead_time_value = lead_time_value - 735963
                        st.info(f"{i}: {lead_time_label} : {abs(corrected_lead_time_value)} days")
                else:
                    st.warning("No lead time suggestions found for the selected date. Please try a different date.")
            except ValueError:
                st.error("Please enter valid numeric values for year, month, day, and selected day.")

    if submit_button:
        if not all(
                [hotel, meal, market_segment, distribution_channel, reserved_room_type, deposit_type, customer_type]):
            st.error("Please fill in all required fields.")
        else:
            hotel_mapping = {'Resort Hotel': 0, 'City Hotel': 1}
            meal_mapping = {'Bed & Breakfast': 0, 'Full Board': 1, 'Half Board': 2, 'Undefined/SC': 3}
            market_segment_mapping = {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                      'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7}
            distribution_channel_mapping = {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4}
            reserved_room_type_mapping = {'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8}
            deposit_type_mapping = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3}
            customer_type_mapping = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
            is_repeated_guest_mapping = {'No': 0, 'Yes': 1}

            hotel_numeric = hotel_mapping.get(hotel)
            meal_numeric = meal_mapping.get(meal)
            market_segment_numeric = market_segment_mapping.get(market_segment)
            distribution_channel_numeric = distribution_channel_mapping.get(distribution_channel)
            reserved_room_type_numeric = reserved_room_type_mapping.get(reserved_room_type)
            deposit_type_numeric = deposit_type_mapping.get(deposit_type)
            customer_type_numeric = customer_type_mapping.get(customer_type)
            is_repeated_guest_numeric = is_repeated_guest_mapping.get(is_repeated_guest)
            try:

                # Convert lead_time, children, agent, company, adr to normalized values
                lead_time_numeric = normalize_value(float(lead_time))
                children_numeric = normalize_value(float(children))
                agent_numeric = normalize_value(float(agent))
                company_numeric = normalize_value(float(company))
                adr_numeric = normalize_value(float(adr))
                month_number_numeric = normalize_value(float(month_number))
                selected_day_numeric = normalize_value(float(selected_day))

                # Ensure numeric fields are validated
                numeric_fields = {
                    "stays_in_weekend_nights": stays_in_weekend_nights,
                    "stays_in_week_nights": stays_in_week_nights,
                    "adults": adults,
                    "babies": babies,
                    "previous_cancellations": previous_cancellations,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled,
                    "required_car_parking_spaces": required_car_parking_spaces,
                    "total_of_special_requests": total_of_special_requests
                }
                # Fill in missing values or set default values
                for field, value in numeric_fields.items():
                    if not isinstance(value, (int, float)):
                        value = float(value)
                    # Add required_car_parking_spaces if not present in input form
                    if field == "required_car_parking_spaces" and value is None:
                        value = 0

                # Prepare data for prediction
                data = np.array([[hotel_numeric, meal_numeric, market_segment_numeric, distribution_channel_numeric,
                                  reserved_room_type_numeric, deposit_type_numeric, customer_type_numeric,
                                  year, month, day, lead_time_numeric, month_number_numeric, selected_day_numeric,
                                  float(stays_in_weekend_nights), float(stays_in_week_nights), float(adults),
                                  children_numeric, float(babies), is_repeated_guest_numeric,
                                  float(previous_cancellations), float(previous_bookings_not_canceled),
                                  agent_numeric, company_numeric, adr_numeric, float(required_car_parking_spaces),
                                  float(total_of_special_requests)]])

                # Print user inputs
                print("User Inputs:")
                print("Hotel:", hotel_numeric)
                print("Meal:", meal_numeric)
                print("Market Segment:", market_segment_numeric)
                print("Distribution Channel:", distribution_channel_numeric)
                print("Reserved Room Type:", reserved_room_type_numeric)
                print("Deposit Type:", deposit_type_numeric)
                print("Customer Type:", customer_type_numeric)
                print("Year:", year)
                print("Month:", month)
                print("Day:", day)
                print("Selected Month:", month_number_numeric)
                print("Selected Day:", selected_day_numeric)
                print("Lead Time:", lead_time_numeric)
                print("Stays in Weekend Nights:", stays_in_weekend_nights)
                print("Stays in Week Nights:", stays_in_week_nights)
                print("Adults:", adults)
                print("Children:", children_numeric)
                print("Babies:", babies)
                print("Is Repeated Guest:", is_repeated_guest_numeric)
                print("Previous Cancellations:", previous_cancellations)
                print("Previous Bookings Not Canceled:", previous_bookings_not_canceled)
                print("Agent:", agent_numeric)
                print("Company:", company_numeric)
                print("ADR:", adr_numeric)
                print("Required Car Parking Spaces:", required_car_parking_spaces)
                print("Total of Special Requests:", total_of_special_requests)

                # Make prediction
                prediction = predict_cancelation(data)

                # Display the results in percentage
                st.success(f"Prediction: {prediction[0][1] * 100:.2f}% probability of cancellation")

            except ValueError as ve:
                st.error(f"Error: {ve}")



            except Exception as e:
                st.error(f"An error occurred while making the prediction: {e}")


#######################################
# Fetch data from the collection and create DataFrame

# Calculate the arrival date from reservation date and lead time
def calculate_arrival_date(row):
    try:
        year = int(row['year'])
        month = int(row['month'])
        day = int(row['day'])
        lead_time = int(row['lead_time'])

        # Filter out unrealistic lead time values
        if lead_time > 5000:
            return pd.NaT

        arrival_date = datetime(year, month, day) + timedelta(days=lead_time)
        return arrival_date
    except Exception as e:
        st.error(f"Error calculating arrival date for row {row.name}: {e}")
        return pd.NaT


# Function to format lead time
def format_lead_time(value):
    if np.isfinite(value):
        return f"{denormalize_value(value)} days"
    else:
        return "N/A"  # Return a placeholder if the value is not finite


# Function to convert encoded columns to human-readable format
def decode_columns(df):
    mappings = {
        'hotel': {0: 'Resort Hotel', 1: 'City Hotel'},
        'meal': {0: 'Bed & Breakfast', 1: 'Full Board', 2: 'Half Board', 3: 'Undefined/SC'},
        'market_segment': {0: 'Direct', 1: 'Corporate', 2: 'Online TA', 3: 'Offline TA/TO', 4: 'Complementary',
                           5: 'Groups', 6: 'Undefined', 7: 'Aviation'},
        'distribution_channel': {0: 'Direct', 1: 'Corporate', 2: 'TA/TO', 3: 'Undefined', 4: 'GDS'},
        'reserved_room_type': {0: 'C', 1: 'A', 2: 'D', 3: 'E', 4: 'G', 5: 'F', 6: 'H', 7: 'L', 8: 'B'},
        'deposit_type': {0: 'No Deposit', 1: 'Refundable', 3: 'Non Refund'},
        'customer_type': {0: 'Transient', 1: 'Contract', 2: 'Transient-Party', 3: 'Group'},
        'year': {i: 2014 + i for i in range(17)}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    return df


def preprocess_features_for_model(df):
    # Encode categorical columns as needed by the model
    categorical_columns = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                           'deposit_type', 'customer_type']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]

    # Ensure all columns expected by the model are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop rows with NaN values

    return df


def insights_screen():
    st.title("Booking Patterns Analysis")

    # Fetch and decode the data
    df = fetch_data()
    if df.empty:
        return
    df = decode_columns(df)

    try:
        # Convert year, month, day, and lead_time to integers
        df['year'] = df['year'].astype(int, errors='ignore')
        df['month'] = df['month'].astype(int, errors='ignore')
        df['day'] = df['day'].astype(int, errors='ignore')
        df['lead_time'] = df['lead_time'].astype(int, errors='ignore')

        # Drop rows with invalid date or lead_time values
        df = df.dropna(subset=['year', 'month', 'day', 'lead_time'])

        df['arrival_date'] = df.apply(calculate_arrival_date, axis=1)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')

        # Drop rows where 'arrival_date' is NaT
        df = df.dropna(subset=['arrival_date'])

        # User selection for hotel type
        hotel_type = st.selectbox("Select Hotel Type", ["Resort Hotel", "City Hotel"])
        filtered_df = df[df['hotel'] == hotel_type]

        if filtered_df.empty:
            st.error(f"No data available for {hotel_type}.")
            return

        # Arrival Dates Distribution
        st.subheader(f"Arrival Dates Distribution for {hotel_type}")
        arrival_dates = filtered_df['arrival_date'].value_counts().reset_index()
        arrival_dates.columns = ['Arrival Date', 'Number of Bookings']
        fig = px.line(arrival_dates, x='Arrival Date', y='Number of Bookings')
        st.plotly_chart(fig)

        # Bookings Distribution by Meal Type
        st.subheader(f"Bookings Distribution by Meal Type for {hotel_type}")
        if 'meal' in filtered_df.columns:
            meal_count = filtered_df['meal'].value_counts().reset_index()
            meal_count.columns = ['Meal Type', 'Number of Bookings']
            fig = px.pie(meal_count, names='Meal Type', values='Number of Bookings')
            st.plotly_chart(fig)
        else:
            st.error("Column 'meal' not found in the dataset.")

        # Market Segment Distribution
        st.subheader(f"Market Segment Distribution for {hotel_type}")
        if 'market_segment' in filtered_df.columns:
            market_segment_count = filtered_df['market_segment'].value_counts().reset_index()
            market_segment_count.columns = ['Market Segment', 'Number of Bookings']
            fig = px.bar(market_segment_count, x='Market Segment', y='Number of Bookings')
            st.plotly_chart(fig)
        else:
            st.error("Column 'market_segment' not found in the dataset.")

        # Average Lead Time by Customer Type
        st.subheader(f"Average Lead Time by Customer Type for {hotel_type}")
        if 'customer_type' in filtered_df.columns and 'lead_time' in filtered_df.columns:
            filtered_df['lead_time'] = pd.to_numeric(filtered_df['lead_time'], errors='coerce')
            lead_time_avg = filtered_df.groupby('customer_type')['lead_time'].mean().reset_index()
            lead_time_avg.columns = ['Customer Type', 'Average Lead Time']
            lead_time_avg['Average Lead Time'] = lead_time_avg['Average Lead Time'].apply(denormalize_value)
            fig = px.bar(lead_time_avg, x='Customer Type', y='Average Lead Time')
            st.plotly_chart(fig)
            st.write(
                "Insight: Customer types with longer lead times tend to plan their bookings well in advance. Consider targeting these customers with early bird offers.")
        else:
            st.error("Columns 'customer_type' or 'lead_time' not found in the dataset.")

        # Booking Trends Over Time
        st.subheader(f"Booking Trends Over Time for {hotel_type}")
        if 'year' in filtered_df.columns and 'month' in filtered_df.columns:
            filtered_df['month'] = pd.to_numeric(filtered_df['month'], errors='coerce')
            booking_trends = filtered_df.groupby(['year', 'month']).size().reset_index(name='Number of Bookings')
            booking_trends['Date'] = pd.to_datetime(booking_trends[['year', 'month']].assign(day=1))
            fig = px.line(booking_trends, x='Date', y='Number of Bookings')
            st.plotly_chart(fig)
            st.write(
                "Insight: Understanding booking trends over time can help in forecasting demand and optimizing pricing strategies.")
        else:
            st.error("Columns 'year' or 'month' not found in the dataset.")

        # Booking Patterns by Market Segment
        st.subheader(f"Booking Patterns by Market Segment for {hotel_type}")
        if 'market_segment' in filtered_df.columns and 'year' in filtered_df.columns:
            filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
            market_segment_trends = filtered_df.groupby(['market_segment', 'year']).size().reset_index(
                name='Number of Bookings')
            market_segment_trends.columns = ['Market Segment', 'Year', 'Number of Bookings']
            fig = px.line(market_segment_trends, x='Year', y='Number of Bookings', color='Market Segment')
            st.plotly_chart(fig)
            st.write(
                "Insight: Tailor your marketing efforts based on which market segments contribute the most to your bookings.")
        else:
            st.error("Columns 'market_segment' or 'year' not found in the dataset.")

        # Guest Satisfaction by Meal Type
        st.subheader(f"Guest Satisfaction by Meal Type for {hotel_type}")
        if 'meal' in filtered_df.columns and 'total_of_special_requests' in filtered_df.columns:
            meal_requests = filtered_df.groupby('meal')['total_of_special_requests'].mean().reset_index()
            meal_requests.columns = ['Meal Type', 'Average Special Requests']
            fig = px.bar(meal_requests, x='Meal Type', y='Average Special Requests')
            st.plotly_chart(fig)
            st.write(
                "Insight: Meal types with higher special requests indicate more demanding guests. Ensure that these requests are met to enhance guest satisfaction.")
        else:
            st.error("Columns 'meal' or 'total_of_special_requests' not found in the dataset.")

        # Lead Time Distribution
        st.subheader(f"Lead Time Distribution for {hotel_type}")
        if 'lead_time' in filtered_df.columns:
            lead_time_distribution = filtered_df['lead_time'].value_counts().reset_index()
            lead_time_distribution.columns = ['Lead Time', 'Number of Bookings']
            lead_time_distribution['Lead Time'] = lead_time_distribution['Lead Time'].apply(denormalize_value)
            fig = px.histogram(lead_time_distribution, x='Lead Time', y='Number of Bookings', nbins=50)
            st.plotly_chart(fig)
            st.write(
                "Insight: Understanding the distribution of lead times can help in planning and managing booking cycles.")
        else:
            st.error("Column 'lead_time' not found in the dataset.")

        # Length of Stay by Customer Type
        st.subheader(f"Length of Stay by Customer Type for {hotel_type}")
        if 'customer_type' in filtered_df.columns and 'stays_in_weekend_nights' in filtered_df.columns and 'stays_in_week_nights' in filtered_df.columns:
            filtered_df['total_stay'] = filtered_df['stays_in_weekend_nights'] + filtered_df['stays_in_week_nights']
            stay_length = filtered_df.groupby('customer_type')['total_stay'].mean().reset_index()
            stay_length.columns = ['Customer Type', 'Average Length of Stay']
            fig = px.bar(stay_length, x='Customer Type', y='Average Length of Stay')
            st.plotly_chart(fig)
            st.write(
                "Insight: Different customer types have varying lengths of stay. This information can be used to tailor marketing campaigns and offers.")
        else:
            st.error(
                "Columns 'customer_type', 'stays_in_weekend_nights', or 'stays_in_week_nights' not found in the dataset.")

        # Cancellation Possibilities for future bookings
        st.subheader("Cancellation Possibilities for Future Bookings")
        today = datetime.today()
        future_bookings = filtered_df[filtered_df['arrival_date'] > today]

        if not future_bookings.empty:
            # Preprocess features for prediction
            features = future_bookings[['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                                        'deposit_type', 'customer_type', 'year', 'month', 'day', 'lead_time',
                                        'arrival_date_week_number', 'arrival_date_day_of_month',
                                        'stays_in_weekend_nights',
                                        'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest',
                                        'previous_cancellations', 'previous_bookings_not_canceled', 'agent', 'company',
                                        'adr',
                                        'required_car_parking_spaces',
                                        'total_of_special_requests']]  # Update with relevant features

            features = preprocess_features_for_model(features)

            # Make predictions using your CatBoost model
            cancellation_probabilities = model.predict_proba(features)[:, 1]
            future_bookings['cancellation_probability'] = cancellation_probabilities

            mean_cancellation_rate = cancellation_probabilities.mean()
            st.write(f"Mean Cancellation Rate for Future Bookings: {mean_cancellation_rate:.2%}")

            # Additional analysis
            st.subheader("Cancellation Rate Distribution for Future Bookings")
            fig = px.histogram(future_bookings, x='cancellation_probability', nbins=10)
            st.plotly_chart(fig)

            st.subheader("Top Factors Influencing Cancellations")
            feature_importance = model.feature_importances_
            features = features.columns
            sorted_indices = np.argsort(feature_importance)[::-1]
            top_features = [(features[i], feature_importance[i]) for i in sorted_indices]
            for feature, importance in top_features[:5]:
                st.write(f"{feature}: {importance:.2f}")

            # Probability of Cancellation for Each Month and Year
            st.subheader("Cancellation Probability by Month")
            future_bookings['month'] = future_bookings['month'].astype(int)  # Ensure month is integer
            monthly_cancellation_prob = future_bookings.groupby('month')[
                'cancellation_probability'].mean().reset_index()
            fig = px.line(monthly_cancellation_prob, x='month', y='cancellation_probability')
            st.plotly_chart(fig)

            st.subheader("Cancellation Probability by Year")
            future_bookings['year'] = future_bookings['year'].astype(int)  # Ensure year is integer
            yearly_cancellation_prob = future_bookings.groupby('year')['cancellation_probability'].mean().reset_index()
            fig = px.line(yearly_cancellation_prob, x='year', y='cancellation_probability')
            st.plotly_chart(fig)

    except Exception as e:

        st.error(f"An error occurred while processing the data: {e}")

#################
## Custom CSS for professional appearance
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #333;
            background: linear-gradient(135deg, #e0f7fa, #d7e4f2);
            margin: 0;
            padding: 0;
        }
        .stTitle {
            font-size: 2.8em;
            text-align: center;
            font-weight: bold;
            margin-bottom: 30px;
            padding: 20px;
            color: white;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            border-radius: 8px;
        }
        .stHeader, .stSubheader {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin: 25px 0 15px 0;
            padding: 15px;
            color: white;
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            border-radius: 8px;
        }
        .stText {
            font-size: 1.3em;
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .stError {
            font-size: 1.3em;
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            color: #e74c3c;
            background: #f9ebea;
            border: 2px solid #e74c3c;
            border-radius: 8px;
        }
        .stNumberInput input {
            font-size: 1.2em;
            text-align: center;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .stData {
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            color: #333;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        .stSection {
            background: #ffffff;
            padding: 25px;
            margin: 25px 0;
            border-radius: 8px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox, .stNumberInput {
            margin: 0 auto;
            text-align: center;
        }
        .center-content {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
    </style>
    """, unsafe_allow_html=True)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import joblib
import pymongo

def optimize_revenue():
    # Load data
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    db = mongo_client["hotel_bookings"]
    collection = db["bookingwithis_canceled"]
    data = collection.find()
    df = pd.DataFrame(list(data))

    # Preprocess data
    df['lead_time'] = np.exp(df['lead_time']) - 1
    df['reservation_year'] = df['year'].apply(lambda x: 2014 + x)
    df['reservation_date'] = df.apply(lambda row: datetime(int(row['reservation_year']), int(row['month']), int(row['day'])), axis=1)
    df['arrival_date'] = df['reservation_date'] + pd.to_timedelta(df['lead_time'], unit='D')
    df['arrival_year'] = df['arrival_date'].dt.year
    df['arrival_month'] = df['arrival_date'].dt.month

    # Load pre-trained model
    model = joblib.load("xgboost_model.joblib")

    # Page Title
    st.markdown("<div class='stTitle'>Optimize Revenue</div>", unsafe_allow_html=True)

    # User inputs section
    st.markdown("<div class='stSection center-content'>", unsafe_allow_html=True)
    st.markdown("<div class='stHeader'>User Inputs</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        hotel_type = st.selectbox("Select Hotel Type", ["Resort Hotel", "City Hotel"], key="hotel_type")
    with col2:
        arrival_year = st.selectbox("Select Arrival Year", list(range(2014, 2028)), key="arrival_year")
    with col3:
        arrival_month = st.selectbox("Select Arrival Month", list(range(1, 13)), key="arrival_month")
    st.markdown("</div>", unsafe_allow_html=True)


    # Encode hotel type
    hotel_type_encoded = 0 if hotel_type == "Resort Hotel" else 1

    # Filter data based on user input
    df_filtered = df[(df['hotel'] == hotel_type_encoded) & (df['arrival_year'] == arrival_year) & (df['arrival_month'] == arrival_month)]
    if df_filtered.empty:
        st.markdown("<div class='stError'>No data available for the selected year, month, and hotel type.</div>", unsafe_allow_html=True)
        return

    # Feature selection for prediction
    features = df_filtered[['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                            'deposit_type', 'customer_type', 'year', 'month', 'day', 'lead_time',
                            'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
                            'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest',
                            'previous_cancellations', 'previous_bookings_not_canceled', 'agent', 'company', 'adr',
                            'required_car_parking_spaces', 'total_of_special_requests']]
    cancellation_probabilities = model.predict_proba(features)[:, 1]
    df_filtered['cancellation_probability'] = cancellation_probabilities

    # Overbooking strategy with dynamic threshold
    dynamic_threshold = np.percentile(cancellation_probabilities, 100 * (1 - (61 / (61 + 10816))))
    high_cancel_prob = df_filtered[df_filtered['cancellation_probability'] >= dynamic_threshold]
    num_high_cancel_prob = len(high_cancel_prob)

    st.markdown(f"<div class='stText'>Out of {len(df_filtered):,} reservations, {num_high_cancel_prob:,} are predicted to be cancelled with a probability >= {dynamic_threshold:.4f}.</div>", unsafe_allow_html=True)

    overbooking_percentage = (num_high_cancel_prob / len(df_filtered)) * 100
    st.markdown(f"<div class='stText'>Recommended overbooking percentage: {overbooking_percentage:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stText'>You can make {num_high_cancel_prob:,} new bookings to compensate for the predicted cancellations.</div>", unsafe_allow_html=True)

    # Visualization
    st.markdown("<div class='stSection center-content'>", unsafe_allow_html=True)
    st.markdown("<div class='stHeader'>Cancellation Probability Distribution</div>", unsafe_allow_html=True)
    fig = px.histogram(df_filtered, x='cancellation_probability', nbins=20, title='Cancellation Probability Distribution')
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Impact analysis for historical data
    if 'is_canceled' in df_filtered.columns:
        st.markdown("<div class='stSection center-content'>", unsafe_allow_html=True)
        st.markdown("<div class='stHeader'>Impact Analysis</div>", unsafe_allow_html=True)
        actual_cancellations = df_filtered[df_filtered['is_canceled'] == 1]
        actual_cancel_rate = len(actual_cancellations) / len(df_filtered) * 100
        st.markdown(f"<div class='stText'>Actual cancellation rate: {actual_cancel_rate:.2f}%</div>", unsafe_allow_html=True)

        predicted_vs_actual = pd.DataFrame({'Type': ['Predicted', 'Actual'], 'Cancellations': [num_high_cancel_prob, len(actual_cancellations)]})
        fig = px.bar(predicted_vs_actual, x='Type', y='Cancellations', title="Predicted vs Actual Cancellations")
        st.plotly_chart(fig)

        overbooking_cost_per_booking = st.number_input("Enter estimated overbooking cost per booking:", value=50, step=1, key="cost_input")
        avg_adr_log = df_filtered['adr'].mean()
        avg_adr = np.exp(avg_adr_log) - 1
        potential_revenue_gain = avg_adr * num_high_cancel_prob
        overbooking_cost = overbooking_cost_per_booking * num_high_cancel_prob
        net_revenue_gain = potential_revenue_gain - overbooking_cost

        st.markdown(f"<div class='stText stData'>Average ADR: ${avg_adr:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Potential revenue gain from overbooking: ${potential_revenue_gain:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Estimated overbooking cost: ${overbooking_cost:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Net revenue gain from overbooking strategy: ${net_revenue_gain:,.2f}</div>", unsafe_allow_html=True)

        st.markdown("<div class='stText'>Note: The net revenue gain from the overbooking strategy assumes that all overbooked reservations are fulfilled without further cancellations.</div>", unsafe_allow_html=True)

        # Compare two scenarios
        actual_revenue_loss = len(actual_cancellations) * avg_adr
        st.markdown(f"<div class='stText'>Scenario 1 - No Model: Actual revenue loss due to cancellations: ${actual_revenue_loss:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText'>Scenario 2 - With Model: Net revenue gain from overbooking strategy: ${net_revenue_gain:,.2f}</div>", unsafe_allow_html=True)

        # Profit/Loss Comparison
        st.markdown("<div class='stSubheader'>Profit/Loss Comparison</div>", unsafe_allow_html=True)
        total_revenue_without_model = len(df_filtered) * avg_adr - actual_revenue_loss
        total_revenue_with_model = len(df_filtered) * avg_adr + net_revenue_gain

        profit_loss_without_model = total_revenue_without_model - actual_revenue_loss
        profit_loss_with_model = total_revenue_with_model - (actual_revenue_loss + overbooking_cost)

        st.markdown(f"<div class='stText stData'>Total revenue without model: ${total_revenue_without_model:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Total revenue with model: ${total_revenue_with_model:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Profit/Loss without model: ${profit_loss_without_model:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText stData'>Profit/Loss with model: ${profit_loss_with_model:,.2f}</div>", unsafe_allow_html=True)

        st.markdown("<div class='stText'>Note: The profit/loss comparison provides a clearer picture of the financial impact of using the model for overbooking strategy.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # For future dates without actual cancellation data
        st.markdown("<div class='stSection center-content'>", unsafe_allow_html=True)
        st.markdown("<div class='stHeader'>Future Date Predictions</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText'>For future dates in {arrival_month}/{arrival_year}, based on historical data and predictions:</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stText'>You can make {num_high_cancel_prob:,} new bookings to compensate for the predicted cancellations.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)















####################################


def edit_customer_details(customer_id):
    header_html = """
        <div style="display: flex; justify-content: center;">
            <h1 style="text-align: center;">Edit Customer Details</h1>
        </div>
    """

    # Display the centered header
    st.write(header_html, unsafe_allow_html=True)

    # Retrieve customer data from the database
    customer_data = collection.find_one({"_id": ObjectId(customer_id)})

    if not customer_data:
        st.error("Customer not found.")
        return

    # Convert internal year format to user-friendly year format
    user_year = internal_to_user_year(customer_data.get("year", 0))

    # Denormalize values for display
    lead_time_denormalized = denormalize_value(customer_data.get("lead_time", 0))
    arrival_date_week_number_denormalized = customer_data.get("arrival_date_week_number", 0)
    arrival_date_day_of_month_denormalized = customer_data.get("arrival_date_day_of_month", 0)
    children_denormalized = denormalize_value(customer_data.get("children", 0))
    agent_denormalized = denormalize_value(customer_data.get("agent", 0))
    company_denormalized = denormalize_value(customer_data.get("company", 0))
    adr_denormalized = denormalize_value(customer_data.get("adr", 0))

    # Function to get key by value in a dictionary
    def get_key_by_value(d, value):
        return list(d.keys())[list(d.values()).index(value)]

    # Reservation form
    with st.form("edit_customer_form"):

        col1, col2 = st.columns(2)

        mappings = {
            "hotel": {'Resort Hotel': 0, 'City Hotel': 1},
            "meal": {'Bed & Breakfast': 0, 'Full Board': 1, 'Half Board': 2, 'Undefined/SC': 3},
            "market_segment": {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                               'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7},
            "distribution_channel": {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4},
            "reserved_room_type": {'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8},
            "deposit_type": {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3},
            "customer_type": {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3},
            "is_repeated_guest": {'No': 0, 'Yes': 1}
        }

        with col1:
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Reservation Details</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            hotel = st.selectbox('Hotel', options=list(mappings["hotel"].keys()),
                                 index=list(mappings["hotel"].values()).index(customer_data.get("hotel", 0)),
                                 key="hotel")
            meal = st.selectbox('Meal', options=list(mappings["meal"].keys()),
                                index=list(mappings["meal"].values()).index(customer_data.get("meal", 0)), key="meal")
            market_segment = st.selectbox('Market Segment', options=list(mappings["market_segment"].keys()),
                                          index=list(mappings["market_segment"].values()).index(
                                              customer_data.get("market_segment", 0)), key="market_segment")
            distribution_channel = st.selectbox('Distribution Channel',
                                                options=list(mappings["distribution_channel"].keys()),
                                                index=list(mappings["distribution_channel"].values()).index(
                                                    customer_data.get("distribution_channel", 0)),
                                                key="distribution_channel")
            reserved_room_type = st.selectbox('Reserved Room Type', options=list(mappings["reserved_room_type"].keys()),
                                              index=list(mappings["reserved_room_type"].values()).index(
                                                  customer_data.get("reserved_room_type", 0)),
                                              key="reserved_room_type")
            deposit_type = st.selectbox('Deposit Type', options=list(mappings["deposit_type"].keys()),
                                        index=list(mappings["deposit_type"].values()).index(
                                            customer_data.get("deposit_type", 0)), key="deposit_type")
            customer_type = st.selectbox('Customer Type', options=list(mappings["customer_type"].keys()),
                                         index=list(mappings["customer_type"].values()).index(
                                             customer_data.get("customer_type", 0)), key="customer_type")
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Date Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            year = st.text_input('Year', value=str(user_year), key="year")
            month = st.slider('Month', 1, 12, customer_data.get("month", 1), key="month")
            day = st.slider('Day', 1, 31, customer_data.get("day", 1), key="day")

            month_options = get_week_options()
            selected_month = st.selectbox('Select Arrival Month', options=month_options, key="selected_month")
            month_number = list(calendar.month_name).index(selected_month)

            day_options = get_day_options(month_number)
            selected_day = st.selectbox('Select Arrival Date Day', options=day_options, key="selected_day")
            lead_time = st.text_input('Lead Time (Press Enter To Get Suggested Lead Times)',
                                      value=str(lead_time_denormalized), key="lead_time")

        with col2:
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Stay Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            is_repeated_guest = st.radio('Is Repeated Guest', options=['No', 'Yes'],
                                         index=customer_data.get("is_repeated_guest", 0), key="is_repeated_guest")
            previous_cancellations = st.text_input('Previous Cancellations',
                                                   value=str(customer_data.get("previous_cancellations", "")),
                                                   key="previous_cancellations")
            previous_bookings_not_canceled = st.text_input('Previous Bookings Not Canceled',
                                                           value=str(
                                                               customer_data.get("previous_bookings_not_canceled", "")),
                                                           key="previous_bookings_not_canceled")
            agent = st.text_input('Agent', value=str(agent_denormalized), key="agent")
            company = st.text_input('Company', value=str(company_denormalized), key="company")
            adr = st.text_input('ADR', value=str(adr_denormalized), key="adr")
            required_car_parking_spaces = st.text_input('Required Car Parking Spaces',
                                                        value=str(customer_data.get("required_car_parking_spaces", "")),
                                                        key="required_car_parking_spaces")
            total_of_special_requests = st.text_input('Total of Special Requests',
                                                      value=str(customer_data.get("total_of_special_requests", "")),
                                                      key="total_of_special_requests")
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Guest Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            stays_in_weekend_nights = st.text_input('Stays in Weekend Nights',
                                                    value=str(customer_data.get("stays_in_weekend_nights", "")),
                                                    key="stays_in_weekend_nights")
            stays_in_week_nights = st.text_input('Stays in Week Nights',
                                                 value=str(customer_data.get("stays_in_week_nights", "")),
                                                 key="stays_in_week_nights")
            adults = st.text_input('Number of Adults', value=str(customer_data.get("adults", "")), key="adults")
            children = st.text_input('Number of Children', value=str(children_denormalized), key="children")
            babies = st.text_input('Number of Babies', value=str(customer_data.get("babies", "")), key="babies")

        submit_button = st.form_submit_button("Update Customer Details")

    get_lead_time_button = st.button('Get Lead Time Suggestions', key="get_lead_time_button")
    if get_lead_time_button:
        if all([year, month, day, selected_month, selected_day]):
            try:
                year = int(year)
                month = int(month)
                day = int(day)
                selected_month = month_number
                selected_day = int(selected_day)

                # Store suggested lead times
                suggested_lead_times = get_lead_time(year, month, day, selected_month, selected_day)
                if suggested_lead_times:  # Check if there are suggested lead times
                    for i, (lead_time_label, lead_time_value) in enumerate(suggested_lead_times, start=1):
                        st.info(f"{i}: {lead_time_label} : {lead_time_value} days")
                else:
                    st.warning("No lead time suggestions found for the selected date. Please try a different date.")
            except ValueError:
                st.error("Please enter valid numeric values for year, month, day, and selected day.")

    if submit_button:
        try:
            # Mapping for categorical columns
            hotel_numeric = mappings["hotel"].get(hotel)
            meal_numeric = mappings["meal"].get(meal)
            market_segment_numeric = mappings["market_segment"].get(market_segment)
            distribution_channel_numeric = mappings["distribution_channel"].get(distribution_channel)
            reserved_room_type_numeric = mappings["reserved_room_type"].get(reserved_room_type)
            deposit_type_numeric = mappings["deposit_type"].get(deposit_type)
            customer_type_numeric = mappings["customer_type"].get(customer_type)
            is_repeated_guest_numeric = mappings["is_repeated_guest"].get(is_repeated_guest)

            # Convert year to internal format
            year_numeric = user_to_internal_year(int(year))

            # Convert lead_time, children, agent, company, adr to normalized values
            lead_time_numeric = normalize_value(float(lead_time))
            children_numeric = normalize_value(float(children))
            agent_numeric = normalize_value(float(agent))
            company_numeric = normalize_value(float(company))
            adr_numeric = normalize_value(float(adr))

            # Update the database with the new values
            numeric_fields = {
                "hotel": hotel_numeric,
                "meal": meal_numeric,
                "market_segment": market_segment_numeric,
                "distribution_channel": distribution_channel_numeric,
                "reserved_room_type": reserved_room_type_numeric,
                "deposit_type": deposit_type_numeric,
                "customer_type": customer_type_numeric,
                "is_repeated_guest": is_repeated_guest_numeric,
                "year": year_numeric,
                "month": int(month),
                "day": int(day),
                "lead_time": lead_time_numeric,
                "arrival_date_week_number": normalize_value(
                    get_week_number(year_numeric, month_number, int(selected_day))),
                "arrival_date_day_of_month": normalize_value(int(selected_day)),
                "stays_in_weekend_nights": float(stays_in_weekend_nights),
                "stays_in_week_nights": float(stays_in_week_nights),
                "adults": float(adults),
                "children": children_numeric,
                "babies": float(babies),
                "previous_cancellations": float(previous_cancellations),
                "previous_bookings_not_canceled": float(previous_bookings_not_canceled),
                "agent": agent_numeric,
                "company": company_numeric,
                "adr": adr_numeric,
                "required_car_parking_spaces": float(required_car_parking_spaces),
                "total_of_special_requests": float(total_of_special_requests),
                "arrival_month": normalize_value(month_number),
                "arrival_day": normalize_value(int(selected_day))
            }

            collection.update_one({"_id": ObjectId(customer_id)}, {"$set": numeric_fields})

            st.success("Customer details updated successfully!")

        except ValueError as ve:
            st.error(f"Error: {ve}")

        except Exception as e:
            st.error(f"An error occurred while updating customer details: {e}")

    st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

    # Delete Customer Button
    # Delete Customer Button
    st.write("<h3 style='text-align:center;'>Delete Customer</h3>", unsafe_allow_html=True)

    delete_button = st.button("Delete Customer")
    confirm_delete = st.checkbox("I confirm to delete this customer")

    if delete_button and confirm_delete:
        try:
            collection.delete_one({"_id": ObjectId(customer_id)})
            st.success(f"Customer with ID {customer_id} has been successfully deleted.")
        except Exception as e:
            st.error(f"An error occurred while deleting the customer: {e}")
    elif delete_button and not confirm_delete:
        st.warning("Please confirm the deletion before proceeding.")


#################

def reservation_page():
    header_html = """
        <div style="display: flex; justify-content: center;">
            <h1 style="text-align: center;">Hotel Reservation</h1>
        </div>
    """
    st.write(header_html, unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {}

    # Restore form values from session state
    form_values = st.session_state.form_values

    with st.form("reservation_form", clear_on_submit=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.image("hotelreservation.png", use_column_width=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input('First Name', value=form_values.get('name', ''))
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Reservation Details</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            hotel = st.selectbox('Hotel', options=['Resort Hotel', 'City Hotel'], index=form_values.get('hotel', 0))
            meal = st.selectbox('Meal', options=['Bed & Breakfast', 'Full Board', 'Half Board', 'Undefined/SC'],
                                index=form_values.get('meal', 0))
            market_segment = st.selectbox('Market Segment',
                                          options=['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary',
                                                   'Groups', 'Undefined', 'Aviation'],
                                          index=form_values.get('market_segment', 0))
            distribution_channel = st.selectbox('Distribution Channel',
                                                options=['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'],
                                                index=form_values.get('distribution_channel', 0))
            reserved_room_type = st.selectbox('Reserved Room Type',
                                              options=['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'B'],
                                              index=form_values.get('reserved_room_type', 0))
            deposit_type = st.selectbox('Deposit Type', options=['No Deposit', 'Refundable', 'Non Refund'],
                                        index=form_values.get('deposit_type', 0))
            customer_type = st.selectbox('Customer Type', options=['Transient', 'Contract', 'Transient-Party', 'Group'],
                                         index=form_values.get('customer_type', 0))
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Date Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            year = st.text_input('Year (The Year the Reservation was Made or Updated)',
                                 value=form_values.get('year', ''))
            month = st.slider('Month (The Month the Reservation was Made or Updated)', 1, 12,
                              value=form_values.get('month', 1))
            day = st.slider('Day (The Day of the Month the Reservation was Made or Updated)', 1, 31,
                            value=form_values.get('day', 1))

            month_options = get_week_options()
            selected_month = st.selectbox('Select Arrival Month', options=month_options,
                                          index=form_values.get('selected_month_index', 0))
            month_number = list(calendar.month_name).index(selected_month)

            day_options = get_day_options(month_number)
            selected_day = st.selectbox('Select Arrival Date Day', options=day_options,
                                        index=form_values.get('selected_day', 0))
            lead_time = st.text_input('Lead Time', value=form_values.get('lead_time', ''))

        with col2:
            surname = st.text_input('Last Name', value=form_values.get('surname', ''))
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Stay Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            is_repeated_guest = st.radio('Is Repeated Guest', options=['No', 'Yes'],
                                         index=form_values.get('is_repeated_guest', 0))
            previous_cancellations = st.text_input('Previous Cancellations',
                                                   value=form_values.get('previous_cancellations', '0'))
            previous_bookings_not_canceled = st.text_input('Previous Bookings Not Canceled',
                                                           value=form_values.get('previous_bookings_not_canceled', '0'))
            agent = st.text_input('Agent', value=form_values.get('agent', '0'))
            company = st.text_input('Company', value=form_values.get('company', '0'))
            adr = st.text_input('ADR', value=form_values.get('adr', ''))
            required_car_parking_spaces = st.text_input('Required Car Parking Spaces',
                                                        value=form_values.get('required_car_parking_spaces', '0'))
            total_of_special_requests = st.text_input('Total of Special Requests',
                                                      value=form_values.get('total_of_special_requests', '0'))
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            st.write("<h2 style='text-align: center;'>Guest Information</h2>", unsafe_allow_html=True)
            st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
            stays_in_weekend_nights = st.text_input('Stays in Weekend Nights',
                                                    value=form_values.get('stays_in_weekend_nights', ''))
            stays_in_week_nights = st.text_input('Stays in Week Nights',
                                                 value=form_values.get('stays_in_week_nights', '1'))
            adults = st.text_input('Number of Adults', value=form_values.get('adults', '1'))
            children = st.text_input('Number of Children', value=form_values.get('children', ''))
            babies = st.text_input('Number of Babies', value=form_values.get('babies', ''))
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

            get_lead_time_button = st.form_submit_button("Get Lead Time Suggestions")
        submit_button = st.form_submit_button("Submit reservation")

    if get_lead_time_button:
        # Update session state with current form values
        st.session_state.form_values = {
            'name': name,
            'surname': surname,
            'hotel': ['Resort Hotel', 'City Hotel'].index(hotel),
            'meal': ['Bed & Breakfast', 'Full Board', 'Half Board', 'Undefined/SC'].index(meal),
            'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups',
                               'Undefined', 'Aviation'].index(market_segment),
            'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'].index(distribution_channel),
            'reserved_room_type': ['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'B'].index(reserved_room_type),
            'deposit_type': ['No Deposit', 'Refundable', 'Non Refund'].index(deposit_type),
            'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'].index(customer_type),
            'year': year,
            'month': month,
            'day': day,
            'selected_month_index': month_number - 1,
            'selected_day': selected_day,
            'lead_time': lead_time,
            'is_repeated_guest': ['No', 'Yes'].index(is_repeated_guest),
            'previous_cancellations': previous_cancellations,
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'agent': agent,
            'company': company,
            'adr': adr,
            'required_car_parking_spaces': required_car_parking_spaces,
            'total_of_special_requests': total_of_special_requests,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'adults': adults,
            'children': children,
            'babies': babies
        }

        try:
            arrival_month = month_number
            arrival_day = selected_day

            # Validate if the year is a valid integer
            if not year.isdigit():
                st.error("Invalid year. Please enter a valid year.")
                return

            suggested_lead_times = get_lead_time(int(year), int(month), int(day), arrival_month, arrival_day)

            if not suggested_lead_times:
                st.error("No valid lead times found. Please check the input dates and try again.")
            else:
                st.write("## Suggested Lead Times:")
                for label, lead_time in suggested_lead_times:
                    st.info(f"{label}: {lead_time} days")
        except Exception as e:
            st.error(str(e))

    if submit_button:
        if not all([hotel, meal, market_segment, distribution_channel, reserved_room_type, deposit_type,
                    customer_type, name, surname]):
            st.error("Please fill in all required fields.")
        else:
            hotel_mapping = {'Resort Hotel': 0, 'City Hotel': 1}
            meal_mapping = {'Bed & Breakfast': 0, 'Full Board': 1, 'Half Board': 2, 'Undefined/SC': 3}
            market_segment_mapping = {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                      'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7}
            distribution_channel_mapping = {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4}
            reserved_room_type_mapping = {'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8}
            deposit_type_mapping = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3}
            customer_type_mapping = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
            is_repeated_guest_mapping = {'No': 0, 'Yes': 1}

            hotel_numeric = hotel_mapping.get(hotel)
            meal_numeric = meal_mapping.get(meal)
            market_segment_numeric = market_segment_mapping.get(market_segment)
            distribution_channel_numeric = distribution_channel_mapping.get(distribution_channel)
            reserved_room_type_numeric = reserved_room_type_mapping.get(reserved_room_type)
            deposit_type_numeric = deposit_type_mapping.get(deposit_type)
            customer_type_numeric = customer_type_mapping.get(customer_type)
            is_repeated_guest_numeric = is_repeated_guest_mapping.get(is_repeated_guest)

            try:
                year = int(year)
                year_numeric = year - 2014
                month_numeric = int(month)
                day_numeric = int(day)
                lead_time_numeric = normalize_value(float(lead_time))
                arrival_date_week_number = get_week_number(year, month_number, int(selected_day))
                arrival_date_day_of_month = normalize_value(float(selected_day))
                stays_in_weekend_nights_numeric = int(stays_in_weekend_nights)
                stays_in_week_nights_numeric = int(stays_in_week_nights)
                adults_numeric = int(adults)
                children_numeric = normalize_value(float(children))
                babies_numeric = int(babies)
                previous_cancellations_numeric = int(previous_cancellations)
                previous_bookings_not_canceled_numeric = int(previous_bookings_not_canceled)
                agent_numeric = normalize_value(float(agent))
                company_numeric = normalize_value(float(company))
                adr_numeric = normalize_value(float(adr))
                required_car_parking_spaces_numeric = int(required_car_parking_spaces)
                total_of_special_requests_numeric = int(total_of_special_requests)

                numeric_fields = {
                    "year": year_numeric,
                    "month": month,
                    "day": day,
                    "lead_time": lead_time_numeric,
                    "arrival_date_week_number": arrival_date_week_number,
                    "arrival_date_day_of_month": arrival_date_day_of_month,
                    "stays_in_weekend_nights": stays_in_weekend_nights_numeric,
                    "stays_in_week_nights": stays_in_week_nights_numeric,
                    "adults": adults_numeric,
                    "children": children_numeric,
                    "babies": babies_numeric,
                    "previous_cancellations": previous_cancellations_numeric,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled_numeric,
                    "agent": agent_numeric,
                    "company": company_numeric,
                    "adr": adr_numeric,
                    "required_car_parking_spaces": required_car_parking_spaces_numeric,
                    "total_of_special_requests": total_of_special_requests_numeric
                }

                for field, value in numeric_fields.items():
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"{field} must be a numeric value.")

                customer_id = str(ObjectId())
                customer_data = {
                    "hotel": hotel_numeric,
                    "meal": meal_numeric,
                    "market_segment": market_segment_numeric,
                    "distribution_channel": distribution_channel_numeric,
                    "reserved_room_type": reserved_room_type_numeric,
                    "deposit_type": deposit_type_numeric,
                    "customer_type": customer_type_numeric,
                    "year": year_numeric,
                    "month": month_numeric,
                    "day": day_numeric,
                    "lead_time": lead_time_numeric,
                    "arrival_date_week_number": normalize_value(arrival_date_week_number),
                    "arrival_date_day_of_month": arrival_date_day_of_month,
                    "stays_in_weekend_nights": stays_in_weekend_nights_numeric,
                    "stays_in_week_nights": stays_in_week_nights_numeric,
                    "adults": adults_numeric,
                    "children": children_numeric,
                    "babies": babies_numeric,
                    "is_repeated_guest": is_repeated_guest_numeric,
                    "previous_cancellations": previous_cancellations_numeric,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled_numeric,
                    "agent": agent_numeric,
                    "company": company_numeric,
                    "adr": adr_numeric,
                    "required_car_parking_spaces": required_car_parking_spaces_numeric,
                    "total_of_special_requests": total_of_special_requests_numeric,
                    "name": name,
                    "surname": surname
                }
                insert_customer_data(customer_id, customer_data)

                # Prediction logic
                exclude_columns = ['name', 'surname', '_id']
                customer_data_filtered = {k: v for k, v in customer_data.items() if k not in exclude_columns}
                input_data = pd.DataFrame([customer_data_filtered])
                prediction = model.predict_proba(input_data)
                cancellation_probability = prediction[0][1] * 100  # Probability of cancellation
                st.info(f"Cancellation Probability: {cancellation_probability:.2f}%")
            except ValueError as e:
                st.error(f"Error: {str(e)}. Please enter valid numeric values for certain fields.")

    # Display all customers in a box
    header_html = """
        <div style="display: flex; justify-content: center;">
            <h1 style="text-align: center;">Select Customer</h1>
        </div>
    """

    # Create customer list and add "None" option
    customers = list(collection.find())
    customer_list = ["None"] + [
        f"{customer.get('name', 'Unknown')} {customer.get('surname', 'Unknown')} ({str(customer['_id'])})" for customer
        in customers]
    selected_customer = st.selectbox("Select Customer", customer_list)

    # Show selected customer details if a customer is selected
    if selected_customer != "None":
        selected_customer_id = selected_customer.split('(')[-1].strip(')')
        edit_customer_details(selected_customer_id)


########################


#################
def main(customer_object_id=None):
    page = st.experimental_get_query_params().get("page", [None])[0]

    # Check if the page parameter is missing or invalid
    if page not in ["Database (Predict)", "Simulation", "Insights", "How Does This Model Work?", "Reservation", "Optimize Revenue"]:
        # Default to the database page
        page = "Database(Predict)"

    # Display the sidebar for page selection
    selected_page = st.sidebar.selectbox("Select Page",
                                         ["Database (Predict)", "Simulation", "Insights",
                                          "How Does This Model Work?", "Reservation", "Optimize Revenue"])

    # Update the page parameter based on the selected page
    if selected_page == "Database (Predict)":
        page = "Database (Predict)"
    elif selected_page == "Simulation":
        page = "Simulation"
    elif selected_page == "Insights":
        page = "Insights"
    elif selected_page == "How Does This Model Work?":
        page = "How Does This Model Work?"
    elif selected_page == "Reservation":
        page = "Reservation"
    elif selected_page == "Optimize Revenue":
        page = "Optimize Revenue"

    # Display the appropriate page based on the page parameter
    if page == "Database (Predict)":
        # Display the database screen
        with st.expander("Database (Predict)"):
            database_screen(customer_object_id)  # Pass customer_object_id here
    elif page == "Simulation":
        # Display the prediction with random values screen
        with st.expander("Simulation"):
            prediction_random_values_screen()
    elif page == "Insights":
        # Display the cancellation insights screen
        with st.expander("Cancellation Insights"):
            insights_screen()
    elif page == "How Does This Model Work?":
        # Display the explanation of how the app works
        with st.expander("How Does This Model Work?"):
            app_work_explanation()
    elif page == "Reservation":
        # Display the reservation page
        with st.expander("Reservation"):
            reservation_page()
    elif page == "Optimize Revenue":
        # Display the reservation page
        with st.expander("Optimize Revenue"):
            optimize_revenue()


def app_work_explanation():
    # Introduction
    st.markdown("<h1 style='text-align: center;'>How Does This Model Work?</h1>", unsafe_allow_html=True)

    st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

    # Page navigation
    page = st.selectbox("",
                        ["Date The Reservation Was Made", "Arrival Date", "Deposit Type", "Lead Time",
                         "Previous Cancellations"],
                        key="page_select",
                        help="Choose a page to explore")
    st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

    if page == "Date The Reservation Was Made":
        # Cancellation rates by month and day
        cancellation_rates = {
            "January": {
                1: 0.786350, 2: 0.149333, 3: 0.274510, 4: 0.409326, 5: 0.443526,
                6: 0.700000, 7: 0.463964, 8: 0.464744, 9: 0.569307, 10: 0.620968,
                11: 0.599265, 12: 0.563574, 13: 0.370000, 14: 0.455357, 15: 0.351145,
                16: 0.701887, 17: 0.305439, 18: 0.911058, 19: 0.604693, 20: 0.701087,
                21: 0.550136, 22: 0.502907, 23: 0.405000, 24: 0.673160, 25: 0.550847,
                26: 0.610000, 27: 0.455128, 28: 0.596552, 29: 0.315961, 30: 0.487273,
                31: 0.618497
            },
            "February": {
                1: 0.725191, 2: 0.728856, 3: 0.686747, 4: 0.430380, 5: 0.395425,
                6: 0.602888, 7: 0.373016, 8: 0.432000, 9: 0.683775, 10: 0.517157,
                11: 0.327354, 12: 0.373860, 13: 0.503846, 14: 0.277778, 15: 0.404255,
                16: 0.440299, 17: 0.372781, 18: 0.359091, 19: 0.270186, 20: 0.329670,
                21: 0.361564, 22: 0.516854, 23: 0.527859, 24: 0.340909, 25: 0.470443,
                26: 0.250000, 27: 0.501618, 28: 0.395181, 29: 0.348624
            },
            "March": {
                1: 0.575658, 2: 0.407625, 3: 0.385057, 4: 0.439739, 5: 0.265734,
                6: 0.286070, 7: 0.512456, 8: 0.404930, 9: 0.432432, 10: 0.384880,
                11: 0.261993, 12: 0.231214, 13: 0.225000, 14: 0.584906, 15: 0.648421,
                16: 0.381232, 17: 0.267606, 18: 0.400517, 19: 0.241007, 20: 0.301939,
                21: 0.496894, 22: 0.340764, 23: 0.333333, 24: 0.290419, 25: 0.383142,
                26: 0.170213, 27: 0.298507, 28: 0.426901, 29: 0.406593, 30: 0.343173,
                31: 0.334270
            },
            "April": {
                1: 0.326923, 2: 0.297491, 3: 0.331445, 4: 0.614525, 5: 0.391026,
                6: 0.354286, 7: 0.255319, 8: 0.271429, 9: 0.205405, 10: 0.318612,
                11: 0.414239, 12: 0.338129, 13: 0.422222, 14: 0.156463, 15: 0.288952,
                16: 0.251938, 17: 0.422594, 18: 0.322785, 19: 0.347328, 20: 0.280374,
                21: 0.478155, 22: 0.382550, 23: 0.287540, 24: 0.334448, 25: 0.359133,
                26: 0.356618, 27: 0.526196, 28: 0.297368, 29: 0.224490, 30: 0.235294
            },
            "May": {
                1: 0.171717, 2: 0.265152, 3: 0.553797, 4: 0.395028, 5: 0.522901,
                6: 0.368421, 7: 0.186770, 8: 0.195187, 9: 0.370253, 10: 0.438650,
                11: 0.305369, 12: 0.324786, 13: 0.290657, 14: 0.344023, 15: 0.170588,
                16: 0.346041, 17: 0.291545, 18: 0.317901, 19: 0.314286, 20: 0.201893,
                21: 0.193182, 22: 0.319088, 23: 0.307692, 24: 0.238532, 25: 0.229551,
                26: 0.149533, 27: 0.409910, 28: 0.304207, 29: 0.216092, 30: 0.280936,
                31: 0.331818
            },
            "June": {
                1: 0.181481, 2: 0.472019, 3: 0.176471, 4: 0.143426, 5: 0.270096,
                6: 0.360502, 7: 0.364706, 8: 0.234375, 9: 0.295775, 10: 0.231760,
                11: 0.159420, 12: 0.253012, 13: 0.296820, 14: 0.370213, 15: 0.376812,
                16: 0.415771, 17: 0.376190, 18: 0.156780, 19: 0.285714, 20: 0.460199,
                21: 0.337302, 22: 0.275081, 23: 0.242623, 24: 0.210884, 25: 0.169725,
                26: 0.324503, 27: 0.338462, 28: 0.422581, 29: 0.501538, 30: 0.420912
            },
            "July": {
                1: 0.209302, 2: 0.689320, 3: 0.338200, 4: 0.394813, 5: 0.262295,
                6: 0.768603, 7: 0.382789, 8: 0.382353, 9: 0.312500, 10: 0.256484,
                11: 0.259701, 12: 0.256881, 13: 0.424508, 14: 0.258675, 15: 0.185393,
                16: 0.240000, 17: 0.224044, 18: 0.371053, 19: 0.255747, 20: 0.241791,
                21: 0.271978, 22: 0.386189, 23: 0.446721, 24: 0.139423, 25: 0.351190,
                26: 0.207237, 27: 0.248252, 28: 0.290123, 29: 0.263158, 30: 0.223058,
                31: 0.325905
            },
            "August": {
                1: 0.303398, 2: 0.306122, 3: 0.245014, 4: 0.343164, 5: 0.239726,
                6: 0.302752, 7: 0.131965, 8: 0.175202, 9: 0.171598, 10: 0.374670,
                11: 0.410194, 12: 0.153846, 13: 0.177778, 14: 0.280749, 15: 0.150838,
                16: 0.157895, 17: 0.390361, 18: 0.235443, 19: 0.180108, 20: 0.145503,
                21: 0.371041, 22: 0.216049, 23: 0.209302, 24: 0.249300, 25: 0.238845,
                26: 0.177515, 27: 0.092025, 28: 0.112272, 29: 0.202247, 30: 0.204268,
                31: 0.385507
            },
            "September": {
                1: 0.230769, 2: 0.347953, 3: 0.164179, 4: 0.309028, 5: 0.207547,
                6: 0.533835, 7: 0.345725, 8: 0.231047, 9: 0.512702, 10: 0.200000,
                11: 0.187879, 12: 0.200730, 13: 0.135593, 14: 0.356000, 15: 0.396226,
                16: 0.248366, 17: 0.316129, 18: 0.132931, 19: 0.167785, 20: 0.427273,
                21: 0.192429, 22: 0.331010, 23: 0.234323, 24: 0.333333, 25: 0.105978,
                26: 0.284790, 27: 0.281081, 28: 0.473868, 29: 0.262821, 30: 0.216867
            },
            "October": {
                1: 0.213043, 2: 0.258555, 3: 0.210970, 4: 0.155932, 5: 0.218519,
                6: 0.239521, 7: 0.444730, 8: 0.321267, 9: 0.177898, 10: 0.313776,
                11: 0.145110, 12: 0.278325, 13: 0.245179, 14: 0.250000, 15: 0.292079,
                16: 0.260976, 17: 0.635731, 18: 0.260450, 19: 0.355499, 20: 0.245614,
                21: 0.874783, 22: 0.332237, 23: 0.307692, 24: 0.197719, 25: 0.192913,
                26: 0.429553, 27: 0.428125, 28: 0.254279, 29: 0.209677, 30: 0.129921,
                31: 0.129568
            },
            "November": {
                1: 0.186813, 2: 0.312253, 3: 0.391534, 4: 0.340249, 5: 0.186869,
                6: 0.296296, 7: 0.299107, 8: 0.198473, 9: 0.387500, 10: 0.228426,
                11: 0.547297, 12: 0.195652, 13: 0.225641, 14: 0.401786, 15: 0.177474,
                16: 0.470588, 17: 0.592965, 18: 0.304762, 19: 0.360731, 20: 0.229839,
                21: 0.481236, 22: 0.156352, 23: 0.425532, 24: 0.385159, 25: 0.857482,
                26: 0.169355, 27: 0.140288, 28: 0.342657, 29: 0.180000, 30: 0.322957
            },
            "December": {
                1: 0.330579, 2: 0.666667, 3: 0.435233, 4: 0.165441, 5: 0.403226,
                6: 0.218935, 7: 0.735294, 8: 0.122850, 9: 0.695015, 10: 0.439024,
                11: 0.234528, 12: 0.581761, 13: 0.628834, 14: 0.608911, 15: 0.534653,
                16: 0.500000, 17: 0.414365, 18: 0.784240, 19: 0.410256, 20: 0.398773,
                21: 0.629268, 22: 0.591489, 23: 0.640693, 24: 0.269231, 25: 0.440860,
                26: 0.333333, 27: 0.431452, 28: 0.262222, 29: 0.373041, 30: 0.286275,
                31: 0.197183
            }
            # Add more months and their respective days
        }

        data_distribution = {
            ("Month 1", 1): 1011, ("Month 1", 2): 375, ("Month 1", 3): 357, ("Month 1", 4): 193, ("Month 1", 5): 363,
            ("Month 1", 6): 440, ("Month 1", 7): 222, ("Month 1", 8): 312, ("Month 1", 9): 202, ("Month 1", 10): 248,
            ("Month 1", 11): 272, ("Month 1", 12): 291, ("Month 1", 13): 200, ("Month 1", 14): 224,
            ("Month 1", 15): 262, ("Month 1", 16): 265, ("Month 1", 17): 239, ("Month 1", 18): 832,
            ("Month 1", 19): 554, ("Month 1", 20): 368, ("Month 1", 21): 369, ("Month 1", 22): 344,
            ("Month 1", 23): 200, ("Month 1", 24): 462, ("Month 1", 25): 236, ("Month 1", 26): 300,
            ("Month 1", 27): 312, ("Month 1", 28): 290, ("Month 1", 29): 307, ("Month 1", 30): 275,
            ("Month 1", 31): 346,
            ("Month 2", 1): 393, ("Month 2", 2): 402, ("Month 2", 3): 332, ("Month 2", 4): 237, ("Month 2", 5): 306,
            ("Month 2", 6): 277, ("Month 2", 7): 252, ("Month 2", 8): 250, ("Month 2", 9): 604, ("Month 2", 10): 408,
            ("Month 2", 11): 223, ("Month 2", 12): 329, ("Month 2", 13): 260, ("Month 2", 14): 378,
            ("Month 2", 15): 376, ("Month 2", 16): 268, ("Month 2", 17): 338, ("Month 2", 18): 220,
            ("Month 2", 19): 322, ("Month 2", 20): 364, ("Month 2", 21): 307, ("Month 2", 22): 267,
            ("Month 2", 23): 341, ("Month 2", 24): 352, ("Month 2", 25): 406, ("Month 2", 26): 324,
            ("Month 2", 27): 309, ("Month 2", 28): 415, ("Month 2", 29): 218,
            ("Month 3", 1): 304, ("Month 3", 2): 341, ("Month 3", 3): 348, ("Month 3", 4): 307, ("Month 3", 5): 286,
            ("Month 3", 6): 402, ("Month 3", 7): 281, ("Month 3", 8): 284, ("Month 3", 9): 333, ("Month 3", 10): 291,
            ("Month 3", 11): 271, ("Month 3", 12): 346, ("Month 3", 13): 360, ("Month 3", 14): 371,
            ("Month 3", 15): 475, ("Month 3", 16): 341, ("Month 3", 17): 284, ("Month 3", 18): 387,
            ("Month 3", 19): 278, ("Month 3", 20): 361, ("Month 3", 21): 322, ("Month 3", 22): 314,
            ("Month 3", 23): 327, ("Month 3", 24): 334, ("Month 3", 25): 261, ("Month 3", 26): 329,
            ("Month 3", 27): 335, ("Month 3", 28): 342, ("Month 3", 29): 364, ("Month 3", 30): 271,
            ("Month 3", 31): 356,
            ("Month 4", 1): 260, ("Month 4", 2): 279, ("Month 4", 3): 353, ("Month 4", 4): 537, ("Month 4", 5): 312,
            ("Month 4", 6): 350, ("Month 4", 7): 329, ("Month 4", 8): 280, ("Month 4", 9): 370, ("Month 4", 10): 317,
            ("Month 4", 11): 309, ("Month 4", 12): 278, ("Month 4", 13): 315, ("Month 4", 14): 294,
            ("Month 4", 15): 353, ("Month 4", 16): 258, ("Month 4", 17): 478, ("Month 4", 18): 316,
            ("Month 4", 19): 262, ("Month 4", 20): 321, ("Month 4", 21): 412, ("Month 4", 22): 298,
            ("Month 4", 23): 313, ("Month 4", 24): 299, ("Month 4", 25): 323, ("Month 4", 26): 272,
            ("Month 4", 27): 439, ("Month 4", 28): 380, ("Month 4", 29): 392, ("Month 4", 30): 289,
            ("Month 5", 1): 297, ("Month 5", 2): 396, ("Month 5", 3): 316, ("Month 5", 4): 362, ("Month 5", 5): 524,
            ("Month 5", 6): 361, ("Month 5", 7): 257, ("Month 5", 8): 374, ("Month 5", 9): 316, ("Month 5", 10): 326,
            ("Month 5", 11): 298, ("Month 5", 12): 351, ("Month 5", 13): 289, ("Month 5", 14): 343,
            ("Month 5", 15): 340, ("Month 5", 16): 341, ("Month 5", 17): 343, ("Month 5", 18): 324,
            ("Month 5", 19): 350, ("Month 5", 20): 317, ("Month 5", 21): 264, ("Month 5", 22): 351,
            ("Month 5", 23): 338, ("Month 5", 24): 327, ("Month 5", 25): 379, ("Month 5", 26): 321,
            ("Month 5", 27): 222, ("Month 5", 28): 309, ("Month 5", 29): 435, ("Month 5", 30): 299,
            ("Month 5", 31): 220,
            ("Month 6", 1): 270, ("Month 6", 2): 411, ("Month 6", 3): 306, ("Month 6", 4): 251, ("Month 6", 5): 311,
            ("Month 6", 6): 319, ("Month 6", 7): 255, ("Month 6", 8): 320, ("Month 6", 9): 355, ("Month 6", 10): 233,
            ("Month 6", 11): 276, ("Month 6", 12): 332, ("Month 6", 13): 283, ("Month 6", 14): 235,
            ("Month 6", 15): 345, ("Month 6", 16): 279, ("Month 6", 17): 420, ("Month 6", 18): 236,
            ("Month 6", 19): 266, ("Month 6", 20): 402, ("Month 6", 21): 252, ("Month 6", 22): 309,
            ("Month 6", 23): 305, ("Month 6", 24): 294, ("Month 6", 25): 218, ("Month 6", 26): 453,
            ("Month 6", 27): 325, ("Month 6", 28): 310, ("Month 6", 29): 325, ("Month 6", 30): 373,
            ("Month 7", 1): 258, ("Month 7", 2): 721, ("Month 7", 3): 411, ("Month 7", 4): 347, ("Month 7", 5): 244,
            ("Month 7", 6): 1102, ("Month 7", 7): 337, ("Month 7", 8): 340, ("Month 7", 9): 368, ("Month 7", 10): 347,
            ("Month 7", 11): 335, ("Month 7", 12): 327, ("Month 7", 13): 457, ("Month 7", 14): 317,
            ("Month 7", 15): 356, ("Month 7", 16): 350, ("Month 7", 17): 366, ("Month 7", 18): 380,
            ("Month 7", 19): 348, ("Month 7", 20): 335, ("Month 7", 21): 364, ("Month 7", 22): 391,
            ("Month 7", 23): 488, ("Month 7", 24): 416, ("Month 7", 25): 336, ("Month 7", 26): 304,
            ("Month 7", 27): 286, ("Month 7", 28): 324, ("Month 7", 29): 380, ("Month 7", 30): 399,
            ("Month 7", 31): 359,
            ("Month 8", 1): 412, ("Month 8", 2): 294, ("Month 8", 3): 351, ("Month 8", 4): 373, ("Month 8", 5): 292,
            ("Month 8", 6): 327, ("Month 8", 7): 341, ("Month 8", 8): 371, ("Month 8", 9): 338, ("Month 8", 10): 379,
            ("Month 8", 11): 412, ("Month 8", 12): 416, ("Month 8", 13): 360, ("Month 8", 14): 374,
            ("Month 8", 15): 358, ("Month 8", 16): 399, ("Month 8", 17): 415, ("Month 8", 18): 395,
            ("Month 8", 19): 372, ("Month 8", 20): 378, ("Month 8", 21): 442, ("Month 8", 22): 324,
            ("Month 8", 23): 301, ("Month 8", 24): 357, ("Month 8", 25): 381, ("Month 8", 26): 338,
            ("Month 8", 27): 326, ("Month 8", 28): 383, ("Month 8", 29): 356, ("Month 8", 30): 328,
            ("Month 8", 31): 345,
            ("Month 9", 1): 377, ("Month 9", 2): 342, ("Month 9", 3): 335, ("Month 9", 4): 288, ("Month 9", 5): 318,
            ("Month 9", 6): 399, ("Month 9", 7): 269, ("Month 9", 8): 277, ("Month 9", 9): 433, ("Month 9", 10): 205,
            ("Month 9", 11): 330, ("Month 9", 12): 274, ("Month 9", 13): 295, ("Month 9", 14): 250,
            ("Month 9", 15): 371, ("Month 9", 16): 306, ("Month 9", 17): 310, ("Month 9", 18): 331,
            ("Month 9", 19): 298, ("Month 9", 20): 440, ("Month 9", 21): 317, ("Month 9", 22): 287,
            ("Month 9", 23): 303, ("Month 9", 24): 243, ("Month 9", 25): 368, ("Month 9", 26): 309,
            ("Month 9", 27): 185, ("Month 9", 28): 287, ("Month 9", 29): 312, ("Month 9", 30): 332,
            ("Month 10", 1): 230, ("Month 10", 2): 263, ("Month 10", 3): 237, ("Month 10", 4): 295,
            ("Month 10", 5): 270, ("Month 10", 6): 334, ("Month 10", 7): 389, ("Month 10", 8): 221,
            ("Month 10", 9): 371, ("Month 10", 10): 392, ("Month 10", 11): 317, ("Month 10", 12): 406,
            ("Month 10", 13): 363, ("Month 10", 14): 292, ("Month 10", 15): 202, ("Month 10", 16): 410,
            ("Month 10", 17): 431, ("Month 10", 18): 311, ("Month 10", 19): 391, ("Month 10", 20): 285,
            ("Month 10", 21): 1725, ("Month 10", 22): 304, ("Month 10", 23): 351, ("Month 10", 24): 263,
            ("Month 10", 25): 254, ("Month 10", 26): 291, ("Month 10", 27): 320, ("Month 10", 28): 409,
            ("Month 10", 29): 248, ("Month 10", 30): 254, ("Month 10", 31): 301,
            ("Month 11", 1): 273, ("Month 11", 2): 253, ("Month 11", 3): 189, ("Month 11", 4): 241,
            ("Month 11", 5): 198, ("Month 11", 6): 297, ("Month 11", 7): 224, ("Month 11", 8): 131,
            ("Month 11", 9): 240, ("Month 11", 10): 197, ("Month 11", 11): 296, ("Month 11", 12): 184,
            ("Month 11", 13): 195, ("Month 11", 14): 224, ("Month 11", 15): 293, ("Month 11", 16): 272,
            ("Month 11", 17): 398, ("Month 11", 18): 210, ("Month 11", 19): 219, ("Month 11", 20): 248,
            ("Month 11", 21): 453, ("Month 11", 22): 307, ("Month 11", 23): 282, ("Month 11", 24): 283,
            ("Month 11", 25): 842, ("Month 11", 26): 248, ("Month 11", 27): 278, ("Month 11", 28): 143,
            ("Month 11", 29): 200, ("Month 11", 30): 257,
            ("Month 12", 1): 121, ("Month 12", 2): 270, ("Month 12", 3): 193, ("Month 12", 4): 272,
            ("Month 12", 5): 124, ("Month 12", 6): 169, ("Month 12", 7): 544, ("Month 12", 8): 407,
            ("Month 12", 9): 341, ("Month 12", 10): 205, ("Month 12", 11): 307, ("Month 12", 12): 318,
            ("Month 12", 13): 326, ("Month 12", 14): 202, ("Month 12", 15): 202, ("Month 12", 16): 160,
            ("Month 12", 17): 181, ("Month 12", 18): 533, ("Month 12", 19): 156, ("Month 12", 20): 163,
            ("Month 12", 21): 205, ("Month 12", 22): 235, ("Month 12", 23): 231, ("Month 12", 24): 78,
            ("Month 12", 25): 93, ("Month 12", 26): 156, ("Month 12", 27): 248, ("Month 12", 28): 225,
            ("Month 12", 29): 319, ("Month 12", 30): 255, ("Month 12", 31): 142,
        }

        # Function to determine risk level and color
        def get_risk_level(day, month):
            if month in cancellation_rates:
                if day in cancellation_rates[month]:
                    cancellation_rate = cancellation_rates[month][day]
                    overall_mean_cancellation_rate = 0.3708  # Overall mean cancellation rate
                    if cancellation_rate > overall_mean_cancellation_rate + 0.1:
                        return "High Risk", "#FF6347"  # Red color for high risk
                    elif cancellation_rate > overall_mean_cancellation_rate:
                        return "Moderate Risk", "#FFD700"  # Yellow color for moderate risk
                    else:
                        return "Low Risk", "#32CD32"  # Green color for low risk
                else:
                    return "Data Not Available", "#777"  # Gray color for unavailable data
            else:
                return "Data Not Available", "#777"  # Gray color for unavailable data

        # Apply CSS styling to the slider label for day
        st.markdown(
            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Select Day</p>",
            unsafe_allow_html=True
        )

        # User input for day using a slider
        user_day = st.slider(" ", min_value=1, max_value=31, value=1, step=1)

        # Apply CSS styling to the select box label for month
        st.markdown(
            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Select Month</p>",
            unsafe_allow_html=True
        )

        # User input for month using a select box
        user_month = st.selectbox(" ", ["January", "February", "March", "April", "May", "June", "July", "August",
                                        "September", "October", "November", "December"], index=0)

        st.markdown("---")

        # Get risk level and color for user input
        risk_level, color = get_risk_level(user_day, user_month)

        # Display risk assessment with enhanced styling
        st.markdown(
            f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
            f"<h2 style='font-size: 24px; margin-bottom: 10px;'>Risk Assessment</h2>"
            f"<div style='background-color: {color}; color: #fff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #888888;'>"
            f"<p style='font-size: 20px; font-weight: bold; margin-bottom: 0;'>Risk Level</p>"
            f"<p style='font-size: 28px; margin-bottom: 0;'>{risk_level}</p>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Calculate minimum cancellation probability considering feature importance
        def min_cancellation_probability(day, month, importance_day, importance_month):
            if month in cancellation_rates:
                if day in cancellation_rates[month]:
                    day_cancellation_rate = cancellation_rates[month][day]
                    month_cancellation_rate = sum(cancellation_rates[month].values()) / len(cancellation_rates[month])

                    # Calculate weighted cancellation rate using feature importance
                    weighted_cancellation_rate = importance_day * day_cancellation_rate + importance_month * month_cancellation_rate
                    return weighted_cancellation_rate
                else:
                    return "Data Not Available"
            else:
                return "Data Not Available"

        # Importance percentages of features
        importance_month = 21.613808 / 100  # Convert percentage to decimal
        importance_day = 19.880642 / 100  # Convert percentage to decimal

        # Get the minimum cancellation probability considering feature importance
        min_prob_weighted = min_cancellation_probability(user_day, user_month, importance_day, importance_month)

        # Display a styled notification for the minimum cancellation probability
        st.markdown(
            f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
            f"<h3 style='font-size: 20px; color: #333; margin-bottom: 10px;'>Minimum Cancellation Probability</h3>"
            f"<p style='font-size: 18px; color: #666; margin-bottom: 10px;'>When '{user_day}-{user_month}', the minimum cancellation probability is:</p>"
            f"<p style='font-size: 24px; color: #009688; margin-bottom: 5px;'>{min_prob_weighted:.2%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        ###other notif

        # Calculate the percentage of customers with reservations on the selected day
        total_reservations = sum(data_distribution.values())  # Total number of reservations

        # Convert month name to its numerical representation
        month_numerical = {
            "January": "Month 1",
            "February": "Month 2",
            "March": "Month 3",
            "April": "Month 4",
            "May": "Month 5",
            "June": "Month 6",
            "July": "Month 7",
            "August": "Month 8",
            "September": "Month 9",
            "October": "Month 10",
            "November": "Month 11",
            "December": "Month 12",
        }

        selected_month_numerical = month_numerical[user_month]  # Get numerical representation of selected month
        selected_date = (selected_month_numerical, user_day)  # Format the selected date to match the data

        selected_reservations = data_distribution.get(selected_date, 0)  # Number of reservations on the selected day

        if total_reservations > 0:
            percentage = (selected_reservations / total_reservations) * 100
        else:
            percentage = 0

        # Display the notification with enhanced styling
        st.markdown(
            f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
            f"<p style='font-size: 18px; color: #333; margin-bottom: 10px;'>Percentage of Customers with Reservations Updated</p>"
            f"<p style='font-size: 24px; color: #009688; margin-bottom: 5px;'>{percentage:.2f}%</p>"
            f"<p style='font-size: 14px; color: #666; margin-bottom: 0;'>on the {user_day}th day of {user_month}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        ####STARTS
        ####STARTS
        # Explanation for day and month columns
        header_html = """
                <div style="display: flex; justify-content: center;">
                    <h1 style="text-align: center;">Understanding the Impact of 'Day' and 'Month' Columns on Cancellation Rates</h1>
                </div>
            """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Importance order of columns
        # Load feature importance data
        feature_importance_data = {
            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",
                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",
                        "Week Nights", "Market Segment", "Agent", "Special Requests",
                        "Customer Type", "Weekend Nights"],
            "Importance": [21.613808, 19.880642, 19.211161, 18.021702,
                           7.920374, 2.655541, 2.541245, 2.051126,
                           1.932295, 0.989599, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance, placing arrival_date_week_number and arrival_date_day_of_month first
        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue' if feature in ["Month", "Day"] else 'lightgray' for feature in
                  df_feature_importance["Feature"]]
        bars = ax.bar(df_feature_importance["Feature"], df_feature_importance["Importance"], color=colors)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance for Day and Month Columns")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(df_feature_importance["Feature"], rotation=45, ha='right')  # Rotate and align x-axis labels

        ax.grid(False)

        # Annotate the blue sky columns
        for bar, importance, feature in zip(bars, df_feature_importance["Importance"],
                                            df_feature_importance["Feature"]):
            if feature in ["Month", "Day"]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{importance:.2f}", ha='center',
                        va='bottom')

        st.pyplot(fig)

        # Explanation for columns

        explanation = """
                <h6 style='color: black; font-weight: bold;'>Month</h6> 
                This column indicates the month in which the reservation status was updated, holding the highest importance with a feature importance score of approximately 21.61%. This suggests that the month significantly influences the prediction of cancellation.<br><br>
                <h6 style='color: black; font-weight: bold;'>Day</h6> 
                Following closely behind, the day of the month in which the reservation status was updated is the second most important feature, with an importance score of about 19.88%.<br><br>
            """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>{explanation}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        # Additional insights about dates with highest and lowest cancellation rates

        # Data for the top 10 days with the highest cancellation rates
        highest_cancellation_data = {
            'day': [18, 21, 25, 1, 18, 6, 7, 2, 1, 16],
            'month': [1, 10, 11, 1, 12, 7, 12, 2, 2, 1],
            'cancellation_rate': [0.911058, 0.874783, 0.857482, 0.786350, 0.784240, 0.768603, 0.735294, 0.728856,
                                  0.725191, 0.701887]
        }

        # Data for the top 10 days with the lowest cancellation rates
        lowest_cancellation_data = {
            'day': [27, 25, 28, 8, 31, 30, 7, 18, 13, 24],
            'month': [8, 9, 8, 12, 10, 10, 8, 9, 9, 7],
            'cancellation_rate': [0.092025, 0.105978, 0.112272, 0.122850, 0.129568, 0.129921, 0.131965, 0.132931,
                                  0.135593, 0.139423]
        }

        # Month names dictionary
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        # Convert data to DataFrame
        highest_cancellation_df = pd.DataFrame(highest_cancellation_data)
        lowest_cancellation_df = pd.DataFrame(lowest_cancellation_data)

        # Function to create bar plots
        def plot_cancellation_rates(data, title):
            fig, ax = plt.subplots(figsize=(10, 6))
            day_month_labels = [f"{day} {month_names[month]}" for day, month in zip(data['day'], data['month'])]
            bars = ax.barh(day_month_labels, data['cancellation_rate'], color='skyblue')
            ax.set_xlabel('Cancellation Rate', fontsize=14)
            ax.set_ylabel('Day/Month', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.invert_yaxis()  # Invert y-axis to display days with the highest cancellation rate at the top
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', fontsize=10)
            plt.tight_layout()  # Adjust layout to prevent overlap
            return fig

        # Plot the top 10 days with the highest cancellation rates
        header_html = """
                <div style="display: flex; justify-content: center;">
                    <h3 style="text-align: center;">Top 10 Days with the Highest Cancellation Rates</h3>
                </div>
            """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)
        highest_cancellation_fig = plot_cancellation_rates(highest_cancellation_df,
                                                           "")
        st.pyplot(highest_cancellation_fig)

        # Plot the top 10 days with the lowest cancellation rates
        header_html = """
                <div style="display: flex; justify-content: center;">
                    <h3 style="text-align: center;">Top 10 Days with the Lowest Cancellation Rates</h3>
                </div>
            """
        st.write(header_html, unsafe_allow_html=True)
        lowest_cancellation_fig = plot_cancellation_rates(lowest_cancellation_df,
                                                          "")
        st.pyplot(lowest_cancellation_fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>What About These Graphs?</span><br><br>"
            "<b>Dates with Highest Cancellation Rates</b><br>"
            "- There seems to be a concentration of high cancellation rates in the first few days of the year (January 1st and January 16th).<br>"
            "- December also has a couple of days with high cancellation rates, such as December 18th and December 7th.<br><br>"
            "<b>Dates with Lowest Cancellation Rates</b><br>"
            "- August 27th has the lowest cancellation rate at only 9.20%, followed by September 25th and August 28th.<br>"
            "- October and September seem to have several days with low cancellation rates, indicating a trend of lower cancellations during these months.<br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####OTHER

        # Sample cancellation rates by month
        cancellation_rates_by_month = pd.Series({
            'January': 0.567894,
            'February': 0.459169,
            'March': 0.374094,
            'April': 0.346916,
            'May': 0.304956,
            'June': 0.314058,
            'July': 0.361283,
            'August': 0.239011,
            'September': 0.284847,
            'October': 0.372417,
            'November': 0.378452,
            'December': 0.477442
        })

        # Determine the month with the highest and lowest cancellation rates
        highest_cancellation_month = cancellation_rates_by_month.idxmax()
        lowest_cancellation_month = cancellation_rates_by_month.idxmin()

        # Create a table to display the cancellation rates by month
        header_html = """
                <div style="display: flex; justify-content: center;">
                    <h1 style="text-align: center;">Cancellation Rates by Month</h1>
                </div>
            """
        st.write(header_html, unsafe_allow_html=True)
        styled_cancellation_table = cancellation_rates_by_month.reset_index().rename(
            columns={'index': 'Month', 0: 'Cancellation Rate'})
        st.table(styled_cancellation_table.style.format({'Cancellation Rate': '{:.2%}'}))

        # Plot the cancellation rates by month with numbers
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(cancellation_rates_by_month.index, cancellation_rates_by_month.values, color='skyblue')

        # Add text labels to each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Month')
        ax.set_ylabel('Cancellation Rate')
        ax.set_title('Cancellation Rates by Month')
        ax.set_xticklabels(cancellation_rates_by_month.index, rotation=45, ha='right')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        # Additional text
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>Peak Cancellation Months:</span><br>"
            "January, December, and February stand out as the peak cancellation months, with cancellation rates ranging from 45.92% to 56.79%.<br>"
            "These months coincide with the winter season in many regions, suggesting potential factors such as inclement weather, holiday travel plans, or seasonal events influencing cancellation behavior.<br><br>"

            "<span style='color: black; font-weight: bold;'>Mid-range Cancellation Months:</span><br>"
            "March, April, June, July, October, and November exhibit mid-range cancellation rates, ranging from 31.41% to 37.85%.<br>"
            "These months span transitions between seasons, with varying weather conditions and fewer major holidays, potentially resulting in more stable cancellation patterns compared to peak months.<br><br>"

            "<span style='color: black; font-weight: bold;'>Low Cancellation Months:</span><br>"
            "August and September emerge as the months with the lowest cancellation rates, at 23.90% and 28.48%, respectively.<br>"
            "These months typically represent the end of summer and the beginning of autumn, characterized by more predictable travel patterns and fewer disruptions compared to peak holiday seasons.<br><br>"

            "<span style='color: black; font-weight: bold;'>Patterns and Seasonal Trends:</span><br>"
            "<b>1. Holiday Season Impact:</b><br>"
            "- The peak cancellation months of January, December, and February coincide with major holidays such as New Year's Eve, Christmas, and Valentine's Day.<br>"
            "- Increased travel during these holiday periods may lead to higher cancellation rates due to changes in plans, last-minute bookings, or shifts in travel preferences.<br><br>"

            "<b>2. Weather Influence:</b><br>"
            "- Weather conditions can significantly impact travel plans and cancellation behavior.<br>"
            "- Winter months, particularly January and February, may experience higher cancellation rates due to flight cancellations, road closures, or concerns about inclement weather.<br>"
            "- In contrast, summer months like August may see lower cancellation rates as travelers are more confident in their plans with predictable weather conditions.<br><br>"

            "<b>3. Seasonal Events:</b><br>"
            "- Seasonal events such as festivals, conferences, or sporting events can influence cancellation rates.<br>"
            "- Months with significant events, such as June with summer festivals or November with business conferences, may experience fluctuations in cancellation rates as travelers adjust their plans based on event schedules.<br><br>"

            "<b>4. Booking Patterns:</b><br>"
            "- Seasonal booking patterns, such as increased travel during summer vacation or holiday periods, can also impact cancellation rates.<br>"
            "- Understanding these seasonal variations in booking behavior is crucial for predicting and managing cancellations effectively, allowing hotels to optimize their revenue management strategies and enhance guest satisfaction.<br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####PTHER
        # Provided statistics
        mean_cancellation_rate = 0.3479079023413319
        std_dev_cancellation_rate = 0.15107189171875826
        overall_cancellation_rate = 0.370765875346028
        percentage_weekend_cancellations = 0.36622556332706874
        percentage_weekday_cancellations = 0.4000875875875876

        # Display statistics and patterns
        header_html = """
                <div style="display: flex; justify-content: center;">
                    <h1 style="text-align: center;">Cancellation Rate Analysis</h1>
                </div>
            """
        st.write(header_html, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        # Display patterns (Emphasized)
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>What Do These Numbers Mean?</span><br><br>"
            "- The mean cancellation rate across all combinations of days and months is slightly lower than the overall cancellation rate, suggesting that certain days or months might have <b>higher</b> or <b>lower</b> cancellation rates compared to the <b>average</b>.<br>"
            "- The <b>standard deviation</b> of cancellation rates indicates <b>moderate variability</b>, implying that cancellation rates vary to some extent across different days and months.<br>"
            "- The distribution of cancellations between <b>weekends</b> and <b>weekdays</b> shows that cancellations are slightly more prevalent on <b>weekdays</b> compared to <b>weekends</b>, with a <b>higher</b> percentage occurring on <b>weekdays</b>.<br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        stats_table_data = {
            "Statistic": ["Mean Cancellation Rate", "Standard Deviation of Cancellation Rates",
                          "Overall Cancellation Rate", "Percentage of Cancellations on Weekends",
                          "Percentage of Cancellations on Weekdays"],
            "Value": [mean_cancellation_rate, std_dev_cancellation_rate,
                      overall_cancellation_rate, percentage_weekend_cancellations,
                      percentage_weekday_cancellations]
        }

        st.table(stats_table_data)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>Statistics</span><br><br>"
            "<b>Mean Cancellation Rate across all Combinations of Day and Month</b><br>"
            f"This statistic represents the average cancellation rate across all combinations of days and months in the dataset. In this case, it's approximately {mean_cancellation_rate:.2%}, indicating that, on average, about {mean_cancellation_rate:.2%} of hotel reservations across various days and months end up being canceled.<br><br>"

            "<b>Standard Deviation of Cancellation Rates</b><br>"
            "The standard deviation measures the dispersion or variability of cancellation rates across all combinations of days and months. A higher standard deviation suggests greater variability in cancellation rates. Here, the standard deviation is approximately 15.11%, indicating moderate variability in cancellation rates across different days and months.<br><br>"

            "<b>Percentage of Cancellations Occurring on Weekends and Weekdays</b><br>"
            "These percentages represent the proportion of cancellations that occur on weekends and weekdays, respectively. In this case, about 36.62% of cancellations occur on weekends, while 40.01% occur on weekdays.<br>"
            "</div>",
            unsafe_allow_html=True
        )


    elif page == "Arrival Date":

        def determine_risk(week_number, day_of_month, cancellation_rate):
            # Get the cancellation rate for the given week number and day of month

            # Calculate the thresholds based on mean values and importance scores
            high_risk_threshold = week_number_mean * (week_number_importance / 100) + day_of_month_mean * (
                    day_of_month_importance / 100)
            moderate_risk_threshold = high_risk_threshold * 0.8  # Adjust this factor as needed

            if cancellation_rate >= 0.34:
                return "High Risk", "#FF6347"
            elif 0.3 <= cancellation_rate < 0.34:
                return "Moderate Risk", "#FFD700"
            else:
                return "Low Risk", "#32CD32"

        # Real data
        # Real data for Mean Cancellation Rates by Week Number
        mean_cancellation_rates_data = {
            'arrival_date_week_number': list(range(1, 54)),
            'cancellation_rate': [
                0.337799, 0.324836, 0.254173, 0.340067, 0.306859, 0.255474, 0.324929, 0.357595, 0.343291, 0.344071,
                0.298789, 0.303468, 0.365369, 0.367491, 0.388371, 0.405990, 0.422048, 0.449538, 0.360033, 0.437972,
                0.371539, 0.398821, 0.408553, 0.409528, 0.466366, 0.360855, 0.369745, 0.367218, 0.367716, 0.403310,
                0.363271, 0.382769, 0.420861, 0.343534, 0.356784, 0.375346, 0.393258, 0.390602, 0.382855, 0.416945,
                0.402968, 0.390909, 0.382228, 0.328194, 0.392268, 0.375159, 0.223614, 0.250167, 0.373596, 0.396529,
                0.263666, 0.298231, 0.355605
            ]
        }

        # Create DataFrame for mean cancellation rates
        mean_cancellation_rates_df = pd.DataFrame(mean_cancellation_rates_data)

        # Populate the data
        day_of_month_cancellation_rates = [
            0.398895, 0.341638, 0.401352, 0.359309, 0.389559, 0.332024, 0.393931, 0.331185, 0.420005, 0.378973,
            0.378736, 0.327851, 0.383263, 0.353365, 0.398520, 0.340170, 0.378531, 0.362895, 0.417178, 0.368356,
            0.349561, 0.373226, 0.325581, 0.376349, 0.392659, 0.373252, 0.355716, 0.383308, 0.352514, 0.380593,
            0.344359
        ]

        # Mean values and importance scores
        week_number_mean = 27.163376
        day_of_month_mean = 15.798717
        week_number_importance = 19.21
        day_of_month_importance = 18.02

        # Apply CSS styling to the number input label for week number
        st.markdown(
            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Enter Arrival Date Week Number</p>",
            unsafe_allow_html=True
        )

        # User input for week number using a slider
        week_number = st.slider("", min_value=1, max_value=53, value=1)

        if week_number % 4 == 1:
            min_day, max_day = 1, 13
        elif week_number % 4 == 2:
            min_day, max_day = 8, 20
        elif week_number % 4 == 3:
            min_day, max_day = 15, 27
        else:  # week_number % 4 == 0
            min_day, max_day = 22, 31  # Wrap around to the next month

        st.markdown(
            f"<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Enter Arrival Date Day of Month ({min_day}-{max_day})</p>",
            unsafe_allow_html=True
        )

        # User input for day of month within the valid range
        day_of_month = st.slider(f"", min_value=min_day,
                                 max_value=max_day, value=min_day)

        # Calculate cancellation rate
        cancellation_rate = mean_cancellation_rates_df.loc[
            mean_cancellation_rates_df['arrival_date_week_number'] == week_number, 'cancellation_rate'].values[0]

        # Determine risk level based on input values
        risk_level, color = determine_risk(week_number, day_of_month, cancellation_rate)

        # Display risk level to user with color
        st.markdown("---")
        # Display risk assessment with enhanced styling
        st.markdown(
            f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
            f"<h2 style='font-size: 24px; margin-bottom: 10px;'>Risk Assessment</h2>"
            f"<div style='background-color: {color}; color: #fff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #888888;'>"
            f"<p style='font-size: 20px; font-weight: bold; margin-bottom: 0;'>Risk Level</p>"
            f"<p style='font-size: 28px; margin-bottom: 0;'>{risk_level}</p>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
            f"<h3 style='font-size: 20px; color: #333; margin-bottom: 10px;'>Minimum Cancellation Probability</h3>"
            f"<p style='font-size: 18px; color: #666; margin-bottom: 10px;'>For Week Number {week_number} and Day of Month {day_of_month}, the minimum cancellation probability is:</p>"
            f"<p style='font-size: 24px; color: #009688; margin-bottom: 5px;'>{cancellation_rate:.2%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        #####OTHERRRR
        # Load feature importance data
        feature_importance_data = {
            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",
                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",
                        "Week Nights", "Market Segment", "Agent", "Special Requests",
                        "Customer Type", "Weekend Nights"],
            "Importance": [21.613808, 19.880642, 19.211161, 18.021702,
                           7.920374, 2.655541, 2.541245, 2.051126,
                           1.932295, 0.989599, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance, placing arrival_date_week_number and arrival_date_day_of_month first
        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Understanding the Impact of 'Week Number' and 'Day of Month' Columns on Cancellation Rates</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue' if feature in ["Arrival Date Week Number", "Arrival Date Day of Month"] else 'lightgray' for
                  feature in df_feature_importance["Feature"]]
        bars = ax.bar(df_feature_importance["Feature"], df_feature_importance["Importance"], color=colors)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance for Arrival Date Columns")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(df_feature_importance["Feature"], rotation=45, ha='right')  # Rotate and align x-axis labels

        ax.grid(False)

        # Annotate the blue sky columns
        for bar, importance, feature in zip(bars, df_feature_importance["Importance"],
                                            df_feature_importance["Feature"]):
            if feature in ["Arrival Date Week Number", "Arrival Date Day of Month"]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{importance:.2f}", ha='center',
                        va='bottom')

        st.pyplot(fig)

        # Explanation

        st.markdown("<div class='st-ea'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>'Arrival Date Week Number' Column</span><br><br>"
            "The 'Arrival Date Week Number' column holds significant importance in the app, with a feature importance score of approximately 19.21%.<br>"
            "This suggests that the specific week number in which the arrival date falls plays a substantial role in predicting cancellations.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>'Arrival Date Day of Month' Column</span><br><br>"
            "Similarly, the 'Arrival Date Day of Month' column is also important, with an importance score of about 18.02%.<br>"
            "This indicates that the particular day of the month when the arrival date occurs significantly influences cancellation predictions.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        #####OTHERR

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Cancellation Rates by Day of Month</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Data for high cancellation days
        high_cancel_days_data = {
            1: 0.398895, 3: 0.401352, 5: 0.389559, 7: 0.393931, 8: 0.420005, 9: 0.378973, 12: 0.378736,
            14: 0.383263, 15: 0.398520, 16: 0.378531, 17: 0.417178, 22: 0.373226, 24: 0.376349,
            26: 0.392659, 27: 0.373252, 28: 0.383308, 30: 0.380593
        }

        # Data for low cancellation days
        low_cancel_days_data = {
            2: 0.341638, 4: 0.359309, 6: 0.332024, 10: 0.331185, 11: 0.327851, 13: 0.353365, 18: 0.340170,
            19: 0.362895, 20: 0.368356, 21: 0.349561, 23: 0.325581, 25: 0.355716, 29: 0.352514, 31: 0.344359
        }

        # Plot high cancellation days
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(high_cancel_days_data.keys(), high_cancel_days_data.values(), color='red',
               label='High Cancellation Days')
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Cancellation Probability")
        ax.set_title("High Cancellation Days")
        ax.legend()
        # Display values on the graph
        for day, prob in high_cancel_days_data.items():
            ax.text(day, prob, str(day), ha='center', va='bottom')
        st.pyplot(fig)

        # Text for high cancellation days
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>High Cancellation Days</span><br><br>"
            "- Days such as 1st, 3rd, 7th, 8th, and 17th of the month exhibit higher cancellation rates, ranging from approximately 37.32% to 42.00%.<br>"
            "- Days like the 1st and 15th may also correspond to the start and middle of pay cycles for some individuals, potentially impacting travel plans and cancellations.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        # Plot low cancellation days
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(low_cancel_days_data.keys(), low_cancel_days_data.values(), color='green', label='Low Cancellation Days')
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Cancellation Probability")
        ax.set_title("Low Cancellation Days")
        ax.legend()
        # Display values on the graph
        for day, prob in low_cancel_days_data.items():
            ax.text(day, prob, str(day), ha='center', va='bottom')
        st.pyplot(fig)

        # Text for low cancellation days
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>Low Cancellation Days:</span><br><br>"
            "- Conversely, days such as the 2nd, 6th, 10th, and 23rd of the month demonstrate lower cancellation rates, ranging from approximately 32.50% to 36.89%.<br>"
            "- Days toward the end of the month, such as the 25th and 31st, also show relatively lower cancellation rates, which could be attributed to travelers finalizing their plans or facing stricter cancellation policies closer to the arrival date.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)
        ###OTHERRR

        # Data
        data = {
            "arrival_date_day_of_month": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31
            ],
            "April": [
                0.309322, 0.272727, 0.371601, 0.369048, 0.332425, 0.482833, 0.385802, 0.372703, 0.424303, 0.337748,
                0.295385, 0.384880, 0.427208, 0.533958, 0.467866, 0.380665, 0.347432, 0.265306, 0.398119, 0.354331,
                0.490515, 0.512346, 0.326471, 0.399329, 0.347490, 0.528217, 0.443850, 0.484629, 0.483221, 0.326870,
                np.nan
            ],
            "August": [
                0.392430, 0.290801, 0.438247, 0.359897, 0.416122, 0.372703, 0.366525, 0.400000, 0.282857, 0.406504,
                0.404157, 0.340122, 0.443011, 0.501582, 0.409867, 0.378076, 0.421846, 0.386588, 0.391408, 0.334047,
                0.289406, 0.360849, 0.292835, 0.400871, 0.364948, 0.396061, 0.407767, 0.303030, 0.293970, 0.385093,
                0.305851
            ],
            "December": [
                0.476190, 0.334630, 0.346863, 0.371429, 0.380306, 0.363636, 0.351724, 0.587007, 0.488806, 0.172222,
                0.170455, 0.214876, 0.166667, 0.120690, 0.382114, 0.340541, 0.218750, 0.146789, 0.329787, 0.304348,
                0.261194, 0.370558, 0.305263, 0.280952, 0.328571, 0.438406, 0.300595, 0.361446, 0.383648, 0.329810,
                0.321608
            ],
            "February": [
                0.511628, 0.296482, 0.247573, 0.209302, 0.207373, 0.238532, 0.285024, 0.150943, 0.303167, 0.357143,
                0.311688, 0.481707, 0.205333, 0.314961, 0.171674, 0.270531, 0.594758, 0.241706, 0.332461, 0.288256,
                0.366667, 0.187726, 0.336134, 0.347656, 0.397872, 0.400000, 0.219298, 0.554054, 0.406250, np.nan, np.nan
            ],
            "January": [
                0.350877, 0.345304, 0.329730, 0.248555, 0.324138, 0.273292, 0.181818, 0.366864, 0.207692, 0.151724,
                0.158416, 0.299517, 0.285714, 0.480826, 0.241611, 0.203846, 0.150943, 0.344086, 0.270370, 0.263514,
                0.244318, 0.470852, 0.462810, 0.211382, 0.421642, 0.224044, 0.386885, 0.274112, 0.216561, 0.220238,
                0.130841
            ],
            "July": [
                0.354906, 0.434316, 0.331412, 0.314815, 0.337793, 0.396896, 0.474328, 0.340845, 0.405093, 0.271137,
                0.380744, 0.352151, 0.314356, 0.397516, 0.393281, 0.414352, 0.346072, 0.374233, 0.269531, 0.428571,
                0.352113, 0.449782, 0.381743, 0.305419, 0.406190, 0.389408, 0.456290, 0.407303, 0.370572, 0.339367,
                0.328767
            ],
            "June": [
                0.404432, 0.467308, 0.425121, 0.354730, 0.366548, 0.203804, 0.489796, 0.621974, 0.468610, 0.305389,
                0.331230, 0.502475, 0.268000, 0.538660, 0.471933, 0.453453, 0.551786, 0.371429, 0.348534, 0.391421,
                0.387755, 0.283688, 0.464832, 0.290909, 0.296154, 0.353511, 0.457143, 0.414085, 0.374016, 0.410667,
                np.nan
            ],
            "March": [
                0.265152, 0.402020, 0.274590, 0.364548, 0.283439, 0.183857, 0.225455, 0.201878, 0.350427, 0.240283,
                0.327684, 0.261224, 0.315457, 0.279863, 0.391566, 0.256849, 0.312312, 0.371875, 0.312676, 0.343066,
                0.419643, 0.384615, 0.328467, 0.355372, 0.468835, 0.347639, 0.333333, 0.241245, 0.164835, 0.355685,
                0.350333
            ],
            "May": [
                0.400000, 0.283753, 0.457516, 0.469697, 0.469697, 0.429799, 0.343860, 0.360434, 0.297158, 0.400000,
                0.312500, 0.421801, 0.507143, 0.261830, 0.499014, 0.324930, 0.299479, 0.357595, 0.553448, 0.521186,
                0.351585, 0.301829, 0.266212, 0.488706, 0.329327, 0.317287, 0.449782, 0.388199, 0.254545, 0.470726,
                0.447439
            ],
            "November": [
                0.511194, 0.231818, 0.478261, 0.500000, 0.419708, 0.210762, 0.558824, 0.156028, 0.318471, 0.284264,
                0.311828, 0.341232, 0.313514, 0.171271, 0.146341, 0.230392, 0.271967, 0.224868, 0.253886, 0.318996,
                0.141791, 0.201754, 0.298413, 0.229947, 0.298319, 0.248555, 0.239437, 0.256983, 0.325581, 0.229508,
                np.nan
            ],
            "October": [
                0.508197, 0.175141, 0.535452, 0.378299, 0.478155, 0.394850, 0.344920, 0.481229, 0.358491, 0.403888,
                0.302817, 0.335498, 0.470588, 0.199248, 0.367188, 0.559748, 0.501475, 0.234127, 0.274011, 0.308989,
                0.452381, 0.433460, 0.130584, 0.456081, 0.135314, 0.339921, 0.404858, 0.403805, 0.365854, 0.381250,
                0.363905
            ],
            "September": [
                0.334471, 0.403974, 0.405405, 0.290123, 0.463529, 0.188000, 0.437736, 0.471519, 0.447867, 0.401575,
                0.337696, 0.381868, 0.138528, 0.263538, 0.414747, 0.387597, 0.498891, 0.480545, 0.381176, 0.388724,
                0.299539, 0.312925, 0.284281, 0.540059, 0.362791, 0.500000, 0.246073, 0.325103, 0.426621, 0.511202,
                np.nan
            ]
        }

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Cancellation Rates by Season</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        df = pd.DataFrame(data)
        df.set_index("arrival_date_day_of_month", inplace=True)

        # Grouping by season
        spring_months = ["March", "April", "May"]
        summer_months = ["June", "July", "August"]
        fall_months = ["September", "October", "November"]
        winter_months = ["December", "January", "February"]

        spring_data = df[spring_months].mean(axis=1)
        summer_data = df[summer_months].mean(axis=1)
        fall_data = df[fall_months].mean(axis=1)
        winter_data = df[winter_months].mean(axis=1)

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Spring
        axes[0, 0].plot(spring_data.index, spring_data.values, marker='o', color='b')
        axes[0, 0].set_title('Spring')
        axes[0, 0].set_xlabel('Day of Month')
        axes[0, 0].set_ylabel('Mean Cancellation Rate')

        # Summer
        axes[0, 1].plot(summer_data.index, summer_data.values, marker='o', color='r')
        axes[0, 1].set_title('Summer')
        axes[0, 1].set_xlabel('Day of Month')
        axes[0, 1].set_ylabel('Mean Cancellation Rate')

        # Fall
        axes[1, 0].plot(fall_data.index, fall_data.values, marker='o', color='g')
        axes[1, 0].set_title('Fall')
        axes[1, 0].set_xlabel('Day of Month')
        axes[1, 0].set_ylabel('Mean Cancellation Rate')

        # Winter
        axes[1, 1].plot(winter_data.index, winter_data.values, marker='o', color='y')
        axes[1, 1].set_title('Winter')
        axes[1, 1].set_xlabel('Day of Month')
        axes[1, 1].set_ylabel('Mean Cancellation Rate')

        # Adjust layout
        plt.tight_layout()
        # Show plot
        st.pyplot(fig)

        def display_header(header_text):
            header_html = f"""
                <div style="display: flex; justify-content: center;">
                    <h3 style="text-align: center; font-weight: bold; font-size: 1.2em;">{header_text}</h3>
                </div>
            """
            st.write(header_html, unsafe_allow_html=True)

        # Display header for Winter Season
        display_header("Winter Season (December, January, February)")

        # Display text for Winter Season
        winter_text = """
        - <b><span style='color: black;'>Weekday Patterns:</span></b> Weekdays in winter months tend to have lower cancellation rates compared to weekends. Travelers might be more likely to cancel weekend getaways due to changing weather conditions or last-minute changes in plans.
        - <b><span style='color: black;'>Holiday Peaks:</span></b> Days surrounding major holidays, such as Christmas and New Year's Eve, may experience higher cancellation rates as travelers' plans might change due to family gatherings, weather disruptions, or preference for different destinations.
        - <b><span style='color: black;'>January 1st:</span></b> While New Year's Day (January 1st) might see a spike in cancellations due to travel fatigue or changes in plans, it could also mark the beginning of new travel bookings for the year.
        """
        st.markdown(winter_text, unsafe_allow_html=True)

        # Display header for Spring Season
        display_header("Spring Season (March, April, May)")

        # Display text for Spring Season
        spring_text = """
        - <b><span style='color: black;'>Midweek Getaways:</span></b> Tuesdays through Thursdays tend to have lower cancellation rates during the spring months, indicating that midweek travelers are more committed to their plans.
        - <b><span style='color: black;'>Spring Break Peaks:</span></b> Cancellation rates might fluctuate during spring break periods, with families and students adjusting their plans based on factors like weather forecasts, travel restrictions, or last-minute deals.
        - <b><span style='color: black;'>Easter Weekend:</span></b> Cancellation rates may rise during Easter weekend, especially if travelers prioritize family gatherings or religious observances over planned trips.
        """
        st.markdown(spring_text, unsafe_allow_html=True)

        # Display header for Summer Season
        display_header("Summer Season (June, July, August)")

        # Display text for Summer Season
        summer_text = """
        - <b><span style='color: black;'>Weekend Escapes:</span></b> Fridays through Sundays typically witness higher cancellation rates during the summer months as travelers plan weekend getaways or beach vacations.
        - <b><span style='color: black;'>Last-Minute Deals:</span></b> Sundays, in particular, might see an increase in cancellations as travelers reevaluate their options after browsing weekend deals or considering alternate destinations.
        - <b><span style='color: black;'>August Vacation Peaks:</span></b> Mid-August might experience a surge in cancellations as families return from summer vacations before the start of the new school year.
        """
        st.markdown(summer_text, unsafe_allow_html=True)

        # Display header for Fall Season
        display_header("Fall Season (September, October, November)")

        # Display text for Fall Season
        fall_text = """
        - <b><span style='color: black;'>Shoulder Season Trends:</span></b> September and October, known as shoulder months, may have moderate cancellation rates as travelers take advantage of milder weather and offseason rates.
        - <b><span style='color: black;'>Thanksgiving Patterns:</span></b> Cancellation rates might rise before and after Thanksgiving weekend as travelers navigate family commitments, changing weather conditions, or flight availability.
        - <b><span style='color: black;'>Black Friday:</span></b> The day after Thanksgiving, known as Black Friday, could see fluctuations in cancellations as travelers adjust plans based on shopping deals or holiday promotions.
        """
        st.markdown(fall_text, unsafe_allow_html=True)

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        #######OTHER

        # Data for each season
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        mean_cancellation_rates = [37.83, 38.76, 36.88, 33.12]
        total_cancellations = [12343, 14508, 10481, 6867]
        total_bookings = [32626, 37434, 28418, 20732]

        # Define custom colors for each season
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFEB3B']

        # Create bar plot for total cancellations and total bookings
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(seasons, total_cancellations, label='Total Cancellations', color=colors, alpha=0.7)
        ax.bar(seasons, total_bookings, bottom=total_cancellations, label='Total Bookings', color='lightblue',
               alpha=0.7)
        ax.set_ylabel('Count', fontsize=14)

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h3 style="text-align: center;">Total Cancellations and Total Bookings by Season</h3>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        for i, (cancel, book) in enumerate(zip(total_cancellations, total_bookings)):
            ax.text(i, cancel + book + 200, f'{cancel}\n({book})', ha='center', va='bottom', fontsize=12, color='black')
        fig.patch.set_facecolor('#F9F9F9')  # Set background color
        st.pyplot(fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>Seasonal Analysis</span><br><br>"
            "- Spring exhibits a cancellation rate slightly above the overall average. This might be attributed to factors such as changes in weather, school holidays, or seasonal travel patterns.<br>"
            "- Summer experiences a cancellation rate slightly higher than the overall average, indicating that cancellations are relatively common during this season. This could be due to the increased volume of travel and vacations.<br>"
            "- Autumn shows a cancellation rate slightly below the overall average. This might suggest that bookings made during this season are relatively more stable compared to other seasons.<br>"
            "- Winter has the lowest cancellation rate among the seasons, indicating that bookings made during this period are less likely to be canceled. This could be due to various factors, including fewer leisure travels and more stable travel plans during the winter months.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'></span> Overall, while there are fluctuations in cancellation rates across seasons, the differences are relatively minor. However, understanding these seasonal trends can still provide valuable insights for managing bookings, staffing, and resources more effectively throughout the year.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####OTHERRRR

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Cancellation Rates by Week Number</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Data for weekly trend
        week_numbers = list(range(1, 54))
        weekly_counts = [
            1045, 1216, 1318, 1485, 1385, 1507, 2102, 2212, 2109, 2142, 2065, 2076,
            2414, 2264, 2683, 2404, 2803, 2923, 2397, 2781, 2853, 2545, 2619, 2498,
            2661, 2386, 2664, 2843, 2763, 3082, 2739, 3041, 3576, 3039, 2587, 2166,
            2225, 2660, 2578, 2396, 2695, 2750, 2352, 2270, 1940, 1570, 1677, 1495,
            1780, 1498, 933, 1187, 1811
        ]

        # Data for mean cancellation rates by week number
        mean_cancellation_rates = [
            0.337799, 0.324836, 0.254173, 0.340067, 0.306859, 0.255474, 0.324929,
            0.357595, 0.343291, 0.344071, 0.298789, 0.303468, 0.365369, 0.367491,
            0.388371, 0.405990, 0.422048, 0.449538, 0.360033, 0.437972, 0.371539,
            0.398821, 0.408553, 0.409528, 0.466366, 0.360855, 0.369745, 0.367218,
            0.367716, 0.403310, 0.363271, 0.382769, 0.420861, 0.343534, 0.356784,
            0.375346, 0.393258, 0.390602, 0.382855, 0.416945, 0.402968, 0.390909,
            0.382228, 0.328194, 0.392268, 0.375159, 0.223614, 0.250167, 0.373596,
            0.396529, 0.263666, 0.298231, 0.355605
        ]

        # Create plots
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot for weekly trend
        ax[0].plot(week_numbers, weekly_counts, marker='o', color='blue')
        ax[0].set_title('Weekly Trend', fontsize=16)
        ax[0].set_xlabel('Week Number', fontsize=14)
        ax[0].set_ylabel('Weekly Counts', fontsize=14)

        # Plot for mean cancellation rates by week number
        ax[1].plot(week_numbers, mean_cancellation_rates, marker='o', color='red')
        ax[1].set_title('Mean Cancellation Rates by Week Number', fontsize=16)
        ax[1].set_xlabel('Week Number', fontsize=14)
        ax[1].set_ylabel('Mean Cancellation Rate', fontsize=14)

        # Add some space between the plots
        plt.subplots_adjust(hspace=0.5)
        # Display plots
        st.pyplot(fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>Cancellation Rate Analysis</span><br><br>"
            "Cancellation rates vary across different weeks throughout the year, ranging from 22.36% to 46.64%.<br><br>"

            "<b>Peak Season (Weeks 21-39):</b><br>"
            "<i>Summer Peak:</i> Weeks 21 to 39, roughly corresponding to late May through September, exhibit the highest numbers of arrivals. This period aligns with the traditional summer vacation season when families, students, and travelers take advantage of warmer weather and school breaks to plan trips.<br>"
            "<i>Booking Trends:</i> The surge in arrivals during these weeks suggests that hotels and accommodations experience high demand, necessitating early booking and potentially higher room rates.<br><br>"

            "<b>Shoulder Season (Weeks 1-20, 40-48):</b><br>"
            "<i>Spring and Fall Shoulder Months:</i> Weeks 1 to 20 and 40 to 48 represent the shoulder seasons, characterized by moderate to lower numbers of arrivals compared to peak summer weeks. These periods typically include springtime (March to May) and autumn (September to November).<br>"
            "<i>Variability:</i> While these shoulder months may see fewer arrivals compared to peak summer weeks, there can still be fluctuations in demand due to factors such as holidays, festivals, and local events.<br><br>"

            "<b>Off-Peak Season (Weeks 49-52, 53):</b><br>"
            "<i>Winter Off-Peak:</i> Weeks 49 to 52 and week 53 generally represent the off-peak winter season, extending from late November through December and sometimes into early January.<br>"
            "</div>",
            unsafe_allow_html=True
        )


    elif page == "Deposit Type":

        feature_importance_data = {
            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",
                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",
                        "Week Nights", "Market Segment", "Agent", "Special Requests",
                        "Customer Type", "Weekend Nights"],
            "Importance": [21.613808, 19.880642, 19.211161, 18.021702,
                           7.920374, 2.655541, 2.541245, 2.051126,
                           1.932295, 0.989599, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance, placing arrival_date_week_number and arrival_date_day_of_month first
        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        # Frequency Distribution of Deposit Types
        deposit_type_counts = {
            "No Deposit": 104461,
            "Non Refund": 14587,
            "Refundable": 162
        }

        deposit_type_importance = 7.92

        # Cancellation Rate by Deposit Type
        cancellation_rates = {
            "No Deposit": 0.284020,
            "Non Refund": 0.993624,
            "Refundable": 0.222222
        }

        # Total number of customers
        total_customers = 119210

        # Dropdown selection for deposit type
        st.markdown(
            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Select Deposit Type</p>",
            unsafe_allow_html=True
        )

        deposit_type_input = st.selectbox("", ["No Deposit", "Non Refund", "Refundable"])

        # Calculate the percentage of customers with the specified deposit type
        if deposit_type_input in deposit_type_counts:
            deposit_type_percentage = (deposit_type_counts[deposit_type_input] / total_customers) * 100

            # Determine risk level based on cancellation rate and deposit type
            cancellation_rate = cancellation_rates[deposit_type_input]
            if cancellation_rate > 0.5:
                risk_level = "High Risk"
                color = "#FF6347"  # Red color for high risk
            elif cancellation_rate > 0.2:
                risk_level = "Not High Risk"
                color = "#32CD32"  # Yellow color for moderate risk
            else:
                risk_level = "Not High Risk"
                color = "#32CD32"  # Green color for low risk

            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            # Display risk assessment with enhanced styling
            st.markdown(
                f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
                f"<h2 style='font-size: 24px; margin-bottom: 10px;'>Risk Assessment</h2>"
                f"<div style='background-color: {color}; color: #fff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #888888;'>"
                f"<p style='font-size: 20px; font-weight: bold; margin-bottom: 0;'>Risk Level</p>"
                f"<p style='font-size: 28px; margin-bottom: 0;'>{risk_level}</p>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Display a styled notification for the percentage of customers with the specified deposit type
            st.markdown(
                f"<div style='background-color: #f9f9f9; border: 1px solid #ccc; border-radius: 8px; padding: 20px; text-align: center;'>"
                f"<p style='font-size: 18px; color: #333; margin-bottom: 10px;'>Percentage of Customers with '{deposit_type_input}' Deposit Type</p>"
                f"<p style='font-size: 24px; color: #009688; margin-bottom: 5px;'>{deposit_type_percentage:.2f}%</p>"
                f"</div>",
                unsafe_allow_html=True
            )


        else:
            st.warning("Please select a valid Deposit Type.")

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        ####STARTS
        # Explanation for deposit type column
        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Understanding the Impact of 'Deposit Type' Column on Cancellation Rates</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue' if feature in ["Deposit Type"] else 'lightgray' for feature in
                  df_feature_importance["Feature"]]
        bars = ax.bar(df_feature_importance["Feature"], df_feature_importance["Importance"], color=colors)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance for Deposit Type Columns")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(df_feature_importance["Feature"], rotation=45, ha='right')  # Rotate and align x-axis labels

        # Remove grid
        ax.grid(False)

        # Annotate the blue sky columns
        for bar, importance in zip(bars, df_feature_importance["Importance"]):
            if bar.get_facecolor() == (
                    0.5294117647058824, 0.807843137254902, 0.9215686274509803,
                    1.0):  # Check if the bar color is skyblue
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{importance:.2f}", ha='center',
                        va='bottom')

        st.pyplot(fig)

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        # Explanation for the importance score
        # Explanation for the importance score
        explanation = """
        The importance score of 7.92 attributed to the deposit type suggests that it plays a significant role in the hospitality industry. 
        Hotels must carefully evaluate deposit policies to strike a balance between maximizing revenue and enhancing guest satisfaction.
        """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>{explanation}</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>No Deposit</h6> This option allows customers to reserve accommodations without making any upfront payment or with a fully refundable deposit. It offers flexibility and convenience to guests, as they can cancel their reservations without financial penalties."
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Non Refund</h6> Bookings with a non-refundable deposit require customers to pay upfront, with no option for a refund if they cancel. Although these bookings often come with lower prices, guests bear the risk of losing their deposit in case of cancellation."
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Refundable</h6> Customers opting for bookings with refundable deposits can cancel their reservations and receive a refund of their deposit. This option provides guests with flexibility and peace of mind, albeit usually at a slightly higher cost compared to non-refundable bookings."
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####OTHERRR
        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Cancellation Rate by Deposit Type</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Data
        cancellation_data = {
            "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
            "cancellation_rate": [0.284020, 0.993624, 0.222222]
        }
        # Convert data to DataFrame
        df = pd.DataFrame(cancellation_data)

        # Set style
        sns.set_style("whitegrid")

        # Define custom color palette
        colors = sns.color_palette("coolwarm", len(df))

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting bars with rounded corners and gradient colors
        bars = ax.bar(df["deposit_type"], df["cancellation_rate"], color=colors, linewidth=1, edgecolor='black')

        # Remove grid
        ax.grid(False)

        # Add numerical values on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2%}", ha='center', va='bottom', color='black',
                    fontsize=12)

        # Add grid and remove spines
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set labels and title
        ax.set_ylabel("Cancellation Rate")

        # Show plot
        st.pyplot(fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Non Refund</h6>"
            "<p>With a cancellation rate of approximately 99.36%, bookings made with a non-refundable deposit are almost certain to be canceled. This could indicate several scenarios. For instance, customers might be making speculative bookings or booking multiple options with the intention of canceling most of them later. From a hotel management perspective, while non-refundable bookings might seem beneficial due to the upfront payment, the extremely high cancellation rate could lead to revenue volatility and operational challenges in managing inventory.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>No Deposit</h6>"
            "<p>Bookings made without a deposit (or with a deposit that is fully refundable) have a cancellation rate of about 28.40%. This suggests that a significant portion of these bookings are indeed canceled. It could imply a more cautious approach from customers who are not willing to commit fully upfront. Hotels might see this as an opportunity to attract more bookings, but they also need to consider the potential impact on revenue and occupancy if a substantial portion of these bookings get canceled close to the arrival date.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Refundable</h6>"
            "<p>Bookings with refundable deposits have the lowest cancellation rate among the three deposit types, at approximately 22.22%. This indicates that customers who opt for refundable deposits are less likely to cancel their bookings compared to those with non-refundable or no deposits. From a customer perspective, the refundable option provides flexibility and peace of mind, allowing them to adjust their plans if needed without significant financial consequences. For hotels, offering refundable deposits could be a way to attract more bookings while still maintaining a reasonable level of revenue stability.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Frequency Distribution of Deposit Types</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        ##st.subheader("Pie Chart")
        # Data
        data = {
            "Deposit Type": ["No Deposit", "Non Refund", "Refundable"],
            "Frequency": [104461, 14587, 162]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(df["Deposit Type"], df["Frequency"], color=['skyblue', 'salmon', 'lightgreen'])

        # Add labels and title
        ax.set_xlabel("Deposit Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')

        # Annotate bars with numbers
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Remove grid
        ax.grid(False)

        # Show plot
        st.pyplot(fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>No Deposit</h6>"
            "<p>This category represents bookings made without any deposit requirement. With a frequency of 104,461 bookings, it indicates that a significant portion of customers prefer the option of not paying any deposit upfront when making reservations. This could be due to various reasons such as uncertainty in travel plans, preference for flexibility, or a lack of willingness to commit financially at the time of booking.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Refundable Deposits</h6>"
            "<p>Bookings categorized under 'Non Refund' require customers to pay a deposit upfront, which is non-refundable in case of cancellation. Despite the non-refundable nature of the deposit, there are still 14,587 bookings falling into this category. This suggests that there is a segment of customers who are willing to accept the terms of non-refundable deposits, possibly in exchange for lower booking prices or other incentives offered by hotels.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Non-Refundable Deposits</h6>"
            "<p>Bookings classified as 'Refundable' involve deposits that customers can get back if they cancel their reservations within a specified time frame. Although the frequency of such bookings is relatively low compared to the other deposit types, with only 162 instances, it indicates that there is a subset of customers who prioritize flexibility and are willing to pay a refundable deposit to secure their bookings. These customers might be more risk-averse or have uncertain travel plans, and the option of a refundable deposit provides them with peace of mind.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        #####OTHERRRR

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Average Lead Time by Deposit Type</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Data
        data = {
            "Deposit Type": ["No Deposit", "Non Refund", "Refundable"],
            "Average Lead Time (days)": [88.841951, 212.908891, 152.098765]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df["Deposit Type"], df["Average Lead Time (days)"], color=['#4C72B0', '#DD8452', '#55A868'])

        # Add labels and title
        ax.set_xlabel("Deposit Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Lead Time (days)", fontsize=12, fontweight='bold')

        # Add values on top of bars with larger font size
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f"{height:.1f}", ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='black')

        # Remove grid
        ax.grid(False)

        # Customize ticks
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Adjust layout
        plt.tight_layout()

        # Add horizontal line at y=0 for better clarity
        ax.axhline(y=0, color='black', linewidth=1.5)

        # Show plot
        st.pyplot(fig)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>No Deposit</h6>"
            "<p>Customers who opt for bookings without any deposit tend to have a relatively shorter lead time, with an average lead time of approximately 88.84 days. This suggests that a significant portion of customers who prefer no deposit bookings are likely to make their reservations closer to their intended check-in dates.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Refundable Deposits</h6>"
            "<p>Bookings categorized under 'Non Refund' require customers to pay a non-refundable deposit upfront, and they exhibit a significantly longer average lead time of approximately 212.91 days. This indicates that customers who are willing to accept non-refundable deposit terms tend to plan their trips well in advance.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Non-Refundable Deposits</h6>"
            "<p>Customers who choose refundable deposit options have an average lead time of approximately 152.10 days. While not as long as the lead time for non-refundable bookings, it still suggests that these customers tend to plan their trips relatively early. However, compared to non-refundable bookings, they may be more flexible in their planning and prefer the option of a refundable deposit to accommodate any changes or uncertainties in their travel plans that may arise closer to their check-in dates.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

    elif page == "Lead Time":

        # Fetch data from MongoDB

        lead_time_data = list(collection.find({}, {"lead_time": 1, "_id": 0}))

        # Create a DataFrame from the fetched data

        df = pd.DataFrame(lead_time_data)

        # Calculate lead time statistics

        lead_time_stats = df['lead_time'].describe()

        # Calculate the lead time distribution

        lead_time_distribution = df['lead_time'].value_counts().to_dict()

        # Total number of customers

        total_customers = df.shape[0]

        # Define denormalized lead time range for the slider

        min_lead_time = denormalize_value(lead_time_stats["min"])

        max_lead_time = denormalize_value(lead_time_stats["max"])

        # Display a slider for lead time input

        st.markdown(

            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Select Lead Time</p>",

            unsafe_allow_html=True)

        # Let user choose the denormalized version of lead time

        denormalized_lead_time_input = st.slider("Lead Time (days)", min_value=min_lead_time, max_value=max_lead_time,
                                                 step=1.0, value=min_lead_time)

        # Normalize the lead time input back to the model's scale

        lead_time_input_raw = np.log(denormalized_lead_time_input + 1)

        # Fetch a random customer document (for demonstration purposes)

        customer_data = collection.find_one()

        # Replace the lead time with the selected input

        customer_data['lead_time'] = lead_time_input_raw

        # Remove the '_id' field as it's not a feature

        if '_id' in customer_data:
            del customer_data['_id']

        # Create a DataFrame for the customer data

        input_features = pd.DataFrame([customer_data])

        # Predict the cancellation probability using the CatBoost model

        cancellation_probability = model.predict_proba(input_features)[:, 1][0]

        # Display notification for the cancellation probability

        notification_text = f"The predicted cancellation probability for a lead time of {denormalized_lead_time_input} days is: {cancellation_probability:.2f}"

        # Determine the percentage of customers with the closest lead time

        closest_lead_time = min(lead_time_distribution.keys(),
                                key=lambda x: abs(denormalize_value(x) - denormalized_lead_time_input))

        percentage = (lead_time_distribution.get(closest_lead_time,
                                                 lead_time_distribution[closest_lead_time]) / total_customers) * 100

        notification_text1 = f"The percentage of customers with a lead time of {denormalize_value(closest_lead_time)} days is: {percentage:.2f}% of the total customers."

        # Display risk assessment

        st.markdown("---")

        st.markdown("<h2 style='text-align: center;'>Risk Assessment</h2>", unsafe_allow_html=True)

        # Determine risk level based on lead time input and distribution statistics

        if lead_time_input_raw > lead_time_stats["75%"]:

            risk_level = "High Risk"

            color = "#FF6347"  # Red color for high risk

            risk_percentage = ((lead_time_input_raw - lead_time_stats["75%"]) / (
                        lead_time_stats["max"] - lead_time_stats["75%"])) * 100

            notification_text_risk = f"You are in the top {risk_percentage:.2f}% of lead times."

        elif lead_time_input_raw > lead_time_stats["50%"]:

            risk_level = "Moderate Risk"

            color = "#FFD700"  # Yellow color for moderate risk

            notification_text_risk = ""

        else:

            risk_level = "Low Risk"

            color = "#32CD32"  # Green color for low risk

            notification_text_risk = ""

        # Display risk level with color

        st.markdown(

            f"<div style='background-color: {color}; color: #fff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #888888;'>",

            unsafe_allow_html=True)

        st.write(

            f"The risk level for lead time {denormalized_lead_time_input} days is: <span style='font-size: 20px; font-weight: bold;'>{risk_level}</span>",

            unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Display notifications if applicable

        if notification_text1:
            st.info(notification_text1)

        if notification_text_risk:
            st.info(notification_text_risk)

        if notification_text:
            st.info(notification_text)

        st.markdown("---")

        #### Feature Importance Section

        # Define feature importance data (example data, adjust as needed)

        feature_importance_data = {

            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",

                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",

                        "Week Nights", "Market Segment", "Agent", "Special Requests",

                        "Customer Type", "Weekend Nights"],

            "Importance": [21.61, 19.88, 19.21, 18.02,

                           7.92, 2.66, 2.54, 2.05,

                           1.93, 0.99, 0.72, 0.70,

                           0.66, 0.41]

        }

        # Convert data to DataFrame

        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance

        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        # Display feature importance

        st.markdown("<h2 style='text-align: center;'>Feature Importance</h2>", unsafe_allow_html=True)

        st.table(df_feature_importance)

        #### Lead Time vs. Cancellation Probability Section

        lead_time_cancellation_prob_all = {

            0: 0.066571, 1: 0.092308, 2: 0.103148, 3: 0.100275, 4: 0.102339, 5: 0.131798, 6: 0.139889, 7: 0.129421,

            8: 0.195958, 9: 0.220989, 10: 0.227926, 11: 0.209677, 12: 0.257646, 13: 0.204629, 14: 0.223029,
            15: 0.301435,

            16: 0.248672, 17: 0.266212, 18: 0.272521, 19: 0.321933, 20: 0.290707

        }

        st.markdown("<h2 style='text-align: center;'>Cancellation Rates by Lead Time</h2>", unsafe_allow_html=True)

        # Convert to DataFrame

        df_cancellation_prob = pd.DataFrame(

            list(lead_time_cancellation_prob_all.items()), columns=["Lead Time (days)", "Cancellation Probability"])

        # Plot lead time vs. cancellation probability

        fig = px.line(df_cancellation_prob, x="Lead Time (days)", y="Cancellation Probability",

                      title="Lead Time vs. Cancellation Probability",

                      labels={"Lead Time (days)": "Lead Time (days)",
                              "Cancellation Probability": "Cancellation Probability"})

        st.plotly_chart(fig)

        # Explanation of lead time vs. cancellation probability

        explanation = """

        The values in the dictionary represent the probabilities of cancellation based on the lead time. Each value indicates the likelihood that a booking will be canceled given the corresponding lead time.


        For instance, if a booking is made on the same day as the arrival (lead time of 0 days), the probability of cancellation is approximately 6.66%. As the lead time increases, the cancellation probability tends to increase as well. For example, for a lead time of 20 days, the probability of cancellation is approximately 29.07%.

        """

        st.markdown(explanation, unsafe_allow_html=True)

        #### Lead Time Histogram Section

        st.markdown("<h2 style='text-align: center;'>Lead Time Distribution</h2>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(df['lead_time'], bins=50, color='blue', alpha=0.7)

        ax.set_xlabel('Lead Time')

        ax.set_ylabel('Frequency')

        ax.set_title('Distribution of Lead Time')

        st.pyplot(fig)

        st.markdown("---")

        #### Display Descriptive Statistics

        st.markdown("<h2 style='text-align: center;'>Lead Time Statistics</h2>", unsafe_allow_html=True)

        st.table(lead_time_stats)

    elif page == "Previous Cancellations":

        feature_importance_data = {
            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",
                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",
                        "Week Nights", "Market Segment", "Agent", "Special Requests",
                        "Customer Type", "Weekend Nights"],
            "Importance": [21.613808, 19.880642, 19.211161, 18.021702,
                           7.920374, 2.655541, 2.541245, 2.051126,
                           1.932295, 0.989599, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance
        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        # Frequency Distribution of Previous Cancellations
        previous_cancellations_counts = {
            0: 112731,
            1: 6048,
            2: 114,
            3: 65,
            4: 31,
            5: 19,
            6: 22,
            11: 35,
            13: 12,
            14: 14,
            19: 19,
            21: 1,
            24: 48,
            25: 25,
            26: 26
        }

        previous_cancellations_importance = 2.541245

        # Total number of customers
        total_customers = sum(previous_cancellations_counts.values())

        # Dropdown selection for previous cancellations
        st.markdown(
            "<p style='font-size: 20px; font-weight: bold; color: #ffffff; text-align: center; background: linear-gradient(to right, #56CCF2, #2F80ED); padding: 12px 20px; border-radius: 8px;'>Select Previous Cancellations</p>",
            unsafe_allow_html=True
        )

        previous_cancellations_input = st.select_slider("", options=list(previous_cancellations_counts.keys()))

        # Calculate the percentage of customers with the specified number of previous cancellations
        if previous_cancellations_input in previous_cancellations_counts:
            previous_cancellations_percentage = (previous_cancellations_counts[
                                                     previous_cancellations_input] / total_customers) * 100

            # Determine risk level based on previous cancellations
            mean_previous_cancellations = 0.087191
            std_previous_cancellations = 0.844918

            # Determine risk level based on previous cancellations
            if previous_cancellations_input > mean_previous_cancellations + 2 * std_previous_cancellations:
                risk_level = "Very High Risk"
                color = "#FF0000"  # Red color for very high risk
            elif previous_cancellations_input > mean_previous_cancellations + std_previous_cancellations:
                risk_level = "High Risk"
                color = "#FF6347"  # Orange-red color for high risk
            elif previous_cancellations_input > mean_previous_cancellations:
                risk_level = "Moderate Risk"
                color = "#FFD700"  # Yellow color for moderate risk
            else:
                risk_level = "Low Risk"
                color = "#32CD32"

            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

            # Calculate cancellation probability for each number of previous cancellations
            cancellation_probabilities = {
                0: 0.284020,
                1: 0.993624,
                2: 0.222222,
                3: 0.415385,
                4: 0.806452,
                5: 0.684211,
                6: 0.590909,
                11: 0.714286,
                13: 0.750000,
                14: 0.642857,
                19: 0.937500,
                21: 0.947500,
                24: 0.957500,
                25: 0.967500,
                26: 0.987500
            }

            # Find the minimum cancellation probability and its corresponding number of previous cancellations
            min_cancellation_prob = None
            min_cancellation_prev_cancel = None
            for prev_cancel, importance in cancellation_probabilities.items():
                # Get the cancellation probability for this number of previous cancellations
                cancellation_prob = importance

                # Update the minimum cancellation probability and corresponding number of previous cancellations
                if min_cancellation_prob is None or cancellation_prob < min_cancellation_prob:
                    min_cancellation_prob = cancellation_prob
                    min_cancellation_prev_cancel = prev_cancel

            # Calculate the minimum cancellation probability based on user selection
            min_cancellation_prob = previous_cancellations_importance * cancellation_probabilities.get(
                previous_cancellations_input, 0)

            # Display notification for the minimum cancellation probability
            notification_text = f"The minimum cancellation probability when the 'Previous Cancellations' is {previous_cancellations_input} is: {min_cancellation_prob:.2f}"

            # Display risk assessment
            st.markdown("<h2 style='text-align: center;'>Risk Assessment</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background-color: {color}; color: #fff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #888888;'>",
                unsafe_allow_html=True)
            st.write(
                f"The risk level for {previous_cancellations_input} previous cancellations is: <span style='font-size: 20px; font-weight: bold;'>{risk_level}</span>",
                unsafe_allow_html=True)
            st.info(
                f"Percentage of customers with {previous_cancellations_input} previous cancellations: {previous_cancellations_percentage:.2f}%")
            st.info(notification_text)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please select a valid number of previous cancellations.")

        #####STARTT

        # Define the feature importance data for the previous cancellation page
        feature_importance_data_prev_cancel = {
            "Feature": ["Month", "Day", "Arrival Date Week Number", "Arrival Date Day of Month",
                        "Deposit Type", "Previous Cancellations", "Parking Spaces",
                        "Week Nights", "Market Segment", "Agent", "Special Requests",
                        "Customer Type", "Weekend Nights"],
            "Importance": [21.613808, 19.880642, 19.211161, 18.021702,
                           7.920374, 2.541245, 2.051126,
                           1.932295, 0.989599, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance_prev_cancel = pd.DataFrame(feature_importance_data_prev_cancel)

        # Sort DataFrame by Importance
        df_feature_importance_prev_cancel = df_feature_importance_prev_cancel.sort_values(by="Importance",
                                                                                          ascending=False)

        # Explanation for Previous Cancellations column
        header_html_prev_cancel = """
                    <div style="display: flex; justify-content: center;">
                        <h1 style="text-align: center;">Understanding the Impact of 'Previous Cancellations' on Cancellation Rates</h1>
                    </div>
                """

        # Display the centered header
        st.write(header_html_prev_cancel, unsafe_allow_html=True)

        # Plot feature importance for previous cancellation page
        fig_prev_cancel, ax_prev_cancel = plt.subplots(figsize=(10, 6))
        colors_prev_cancel = ['skyblue' if feature in ["Previous Cancellations"] else 'lightgray' for feature in
                              df_feature_importance_prev_cancel["Feature"]]
        bars_prev_cancel = ax_prev_cancel.bar(df_feature_importance_prev_cancel["Feature"],
                                              df_feature_importance_prev_cancel["Importance"], color=colors_prev_cancel)
        ax_prev_cancel.set_xlabel("Feature")
        ax_prev_cancel.set_ylabel("Importance")
        ax_prev_cancel.set_title("Feature Importance for Previous Cancellations Columns")
        ax_prev_cancel.tick_params(axis='x', rotation=45)
        ax_prev_cancel.set_xticklabels(df_feature_importance_prev_cancel["Feature"], rotation=45,
                                       ha='right')  # Rotate and align x-axis labels

        # Remove grid
        ax_prev_cancel.grid(False)

        # Annotate the blue sky columns
        for bar_prev_cancel, importance_prev_cancel in zip(bars_prev_cancel,
                                                           df_feature_importance_prev_cancel["Importance"]):
            if bar_prev_cancel.get_facecolor() == (
                    0.5294117647058824, 0.807843137254902, 0.9215686274509803,
                    1.0):  # Check if the bar color is skyblue
                ax_prev_cancel.text(bar_prev_cancel.get_x() + bar_prev_cancel.get_width() / 2,
                                    bar_prev_cancel.get_height() + 0.05, f"{importance_prev_cancel:.2f}", ha='center',
                                    va='bottom')

        st.pyplot(fig_prev_cancel)

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        explanation_prev_cancel = """
                    The importance score of 2.54 attributed to the previous cancellations indicates its significance in predicting 
                    cancellation rates. Previous cancellations provide insights into the behavior of guests who have previously 
                    canceled their bookings. It can indicate patterns such as seasonal fluctuations, booking uncertainties, or 
                    dissatisfaction with the service. Understanding the impact of previous cancellations can help hotels implement 
                    targeted strategies to reduce cancellation rates, improve customer satisfaction, and optimize revenue management.
                """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>{explanation_prev_cancel}</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####OTHERR

        header_html = """
                    <div style="display: flex; justify-content: center;">
                        <h1 style="text-align: center;">Distribution of Previous Cancellations</h1>
                    </div>
                """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Distribution of Previous Cancellations Data
        previous_cancellations_data = {
            "previous_cancellations": [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 19, 21, 24, 25, 26],
            "count": [112731, 6048, 114, 65, 31, 19, 22, 35, 12, 14, 19, 1, 48, 25, 26]
        }

        # Convert data to DataFrame
        df_previous_cancellations = pd.DataFrame(previous_cancellations_data)

        # Plot the distribution of previous cancellations
        fig_previous_cancellations, ax_previous_cancellations = plt.subplots(figsize=(10, 6))
        ax_previous_cancellations.bar(df_previous_cancellations["previous_cancellations"],
                                      df_previous_cancellations["count"], color='skyblue')
        ax_previous_cancellations.set_xlabel("Previous Cancellations")
        ax_previous_cancellations.set_ylabel("Count")
        ax_previous_cancellations.grid(False)

        # Annotate the bars with counts
        for i, count in enumerate(df_previous_cancellations["count"]):
            ax_previous_cancellations.text(i, count, str(count), ha='center', va='bottom')

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Show the plot
        st.pyplot(fig_previous_cancellations)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Majority No Previous Cancellations</h6>"
            "<p>Most bookings (112,731 out of 119,210) have no previous cancellations, indicating a lower risk of cancellation for these reservations.</p><br>"
            "<h6 style='color: black; font-weight: bold;'>Decreasing Frequency with More Cancellations</h6>"
            "<p>As the number of previous cancellations increases, the frequency of bookings generally decreases, suggesting that guests with a history of cancellations may book less frequently.</p><br>"
            "<h6 style='color: black; font-weight: bold;'>Some Bookings with High Previous Cancellations</h6>"
            "<p>Although less common, some bookings have high numbers of previous cancellations, which may require special attention to manage risk and ensure reservation reliability.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####other

        header_html = """
                    <div style="display: flex; justify-content: center;">
                        <h1 style="text-align: center;">Cancellation History</h1>
                    </div>
                """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Cancellation History Data
        cancellation_history_data = {
            "Category": ["With Previous Cancellations", "Without Previous Cancellations"],
            "Percentage": [5.43, 100 - 5.43]
            # Assuming the remaining percentage represents bookings without previous cancellations
        }

        # Plot the cancellation history
        fig_cancel_history, ax_cancel_history = plt.subplots()
        ax_cancel_history.pie(cancellation_history_data["Percentage"], labels=cancellation_history_data["Category"],
                              autopct='%1.2f%%', startangle=90, colors=['skyblue', 'lightgray'])
        ax_cancel_history.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Show the plot
        st.pyplot(fig_cancel_history)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Percentage of Bookings with Previous Cancellations</h6>"
            "<p>This percentage represents the proportion of bookings in the dataset that have at least one previous cancellation. In other words, out of all the bookings analyzed, 5.43% of them are associated with guests who have previously canceled at least one reservation.</p><br>"
            "<h6 style='color: black; font-weight: bold;'>Impact on Revenue and Operations</h6>"
            "<p>Analyzing the percentage of bookings with previous cancellations helps hotels assess the potential impact of cancellations on revenue, occupancy, and operational planning. It enables them to implement strategies to mitigate risks associated with cancellations and optimize revenue management practices.</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        #####other

        cancellation_probabilities = {
            0: 0.284020,
            1: 0.993624,
            2: 0.222222,
            3: 0.415385,
            4: 0.806452,
            5: 0.684211,
            6: 0.590909,
            11: 0.714286,
            13: 0.750000,
            14: 0.642857,
            19: 1.000000,
            21: 0.000000,
            24: 0.937500,
            25: 0.800000,
            26: 0.923077
        }

        header_html = """
                    <div style="display: flex; justify-content: center;">
                        <h1 style="text-align: center;">Cancellation Probability</h1>
                    </div>
                """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(cancellation_probabilities.keys(), cancellation_probabilities.values(), color='skyblue')
        plt.xlabel('Number of Previous Cancellations')
        plt.ylabel('Cancellation Probability')
        plt.xticks(list(cancellation_probabilities.keys()))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Display the plot using st.pyplot
        st.pyplot(plt)

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<h6 style='color: black; font-weight: bold;'>Previous Cancellations:</h6>"
            "<p>The keys in the dictionary represent the number of previous cancellations associated with bookings. For example, a key of 0 indicates that there were no previous cancellations, while a key of 1 indicates one previous cancellation, and so on.</p><br>"
            "<h6 style='color: black; font-weight: bold;'>Cancellation Probabilities:</h6>"
            "<p>The values in the dictionary represent the probabilities of future cancellations based on the number of previous cancellations. Each value indicates the likelihood that a booking will be canceled given the corresponding number of previous cancellations.</p><br>"
            "<p>For instance, if a booking has 0 previous cancellations, the probability of cancellation is approximately 28.40%. Similarly, if a booking has 1 previous cancellation, the probability of cancellation dramatically increases to approximately 99.36%.</p><br>"
            "</div>",
            unsafe_allow_html=True)


    elif page == "Stay Duration":
        # Define the feature importance data for the week nights page
        # Load feature importance data
        feature_importance_data = {
            "Feature": ["Month", "Day", "Week Nights", "Weekend Nights",
                        "Deposit Type", "Lead Time", "Previous Cancellations", "Parking Spaces",
                        "Arrival Date Week Number", "Arrival Date Day of Month", "Market Segment", "Agent",
                        "Special Requests",
                        "Customer Type"],
            "Importance": [21.613808, 19.880642, 1.932295, 0.989599,
                           7.920374, 2.655541, 2.541245, 2.051126,
                           19.211161, 18.021702, 0.716995, 0.701381,
                           0.660911, 0.407462]
        }

        # Convert data to DataFrame
        df_feature_importance = pd.DataFrame(feature_importance_data)

        # Sort DataFrame by Importance, placing Week Nights and Weekend Nights first
        df_feature_importance = df_feature_importance.sort_values(by="Importance", ascending=False)

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Understanding the Impact of 'Week Nights' and 'Weekend Nights' Columns on Cancellation Rates</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue' if feature in ["Week Nights", "Weekend Nights"] else 'lightgray' for
                  feature in df_feature_importance["Feature"]]
        bars = ax.bar(df_feature_importance["Feature"], df_feature_importance["Importance"], color=colors)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance for Week Nights and Weekend Nights Columns")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(df_feature_importance["Feature"], rotation=45, ha='right')  # Rotate and align x-axis labels

        ax.grid(False)

        # Annotate the blue sky columns
        for bar, importance, feature in zip(bars, df_feature_importance["Importance"],
                                            df_feature_importance["Feature"]):
            if feature in ["Week Nights", "Weekend Nights"]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{importance:.2f}", ha='center',
                        va='bottom')

        st.pyplot(fig)

        # Explanation

        st.markdown("<div class='st-ea'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>'Week Nights' Column</span><br><br>"
            "The 'Week Nights' column holds significant importance in the app, with a feature importance score of approximately 1.93%.<br>"
            "This suggests that the number of week nights booked significantly influences cancellation predictions.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>"
            "<span style='color: black; font-weight: bold;'>'Weekend Nights' Column</span><br><br>"
            "Similarly, the 'Weekend Nights' column is also important, with an importance score of about 0.99%.<br>"
            "This indicates that the number of weekend nights booked significantly influences cancellation predictions.<br><br>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ###OTHER

        # Data for Total Stay Duration
        total_stay_duration_data = {
            0: 645, 1: 21005, 2: 27632, 3: 27064, 4: 17373,
            5: 7771, 6: 3846, 7: 8648, 8: 1155, 9: 840,
            10: 1135, 11: 393, 12: 220, 13: 141, 14: 913,
            15: 72, 16: 40, 17: 20, 18: 35, 19: 22,
            20: 14, 21: 71, 22: 13, 23: 8, 24: 6,
            25: 37, 26: 6, 27: 4, 28: 34, 29: 13,
            30: 13, 33: 3, 34: 1, 35: 5, 38: 1,
            42: 4, 45: 1, 46: 1, 48: 1, 56: 2,
            60: 1, 69: 1
        }

        # Data for Stays in Week Nights
        stays_in_week_nights_data = {
            0: 7572, 1: 30292, 2: 33670, 3: 22241, 4: 9543,
            5: 11068, 6: 1494, 7: 1024, 8: 654, 9: 228,
            10: 1030, 11: 55, 12: 42, 13: 27, 14: 35,
            15: 85, 16: 15, 17: 4, 18: 6, 19: 43,
            20: 39, 21: 15, 22: 7, 24: 3, 25: 6,
            26: 1, 30: 4, 32: 1, 33: 1, 34: 1,
            40: 2, 42: 1, 50: 1
        }

        # Data for Stays in Weekend Nights
        stays_in_weekend_nights_data = {
            0: 51895, 1: 30615, 2: 33266, 3: 1252, 4: 1847,
            5: 77, 6: 152, 7: 19, 8: 58, 9: 10,
            10: 7, 12: 5, 13: 2, 14: 1, 16: 2,
            18: 1, 19: 1
        }

        # Explanation for Lead Time column
        header_html = """
                                    <div style="display: flex; justify-content: center;">
                                        <h1 style="text-align: center;">Combined Graph of Stay Duration and Nights</h1>
                                    </div>
                                """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Create a new figure
        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot Total Stay Duration
        plt.plot(total_stay_duration_data.keys(), total_stay_duration_data.values(), label='Total Stay Duration')

        # Plot Stays in Week Nights
        plt.plot(stays_in_week_nights_data.keys(), stays_in_week_nights_data.values(), label='Stays in Week Nights')

        # Plot Stays in Weekend Nights
        plt.plot(stays_in_weekend_nights_data.keys(), stays_in_weekend_nights_data.values(),
                 label='Stays in Weekend Nights')

        # Set x-axis range from 0 to 20
        plt.xlim(0, 20)

        # Add labels and title
        plt.xlabel('Duration/Nights')
        plt.ylabel('Count')
        plt.legend()

        # Show the plot in Streamlit
        st.pyplot(plt)
        explanation = """
            <h6 style='color: black; font-weight: bold; text-align: center;'>Total Stay Duration</h6> 
            This dataset provides the count of stays based on their total duration in nights.<br>
            The counts range from stays of 0 nights (possibly indicating same-day cancellations) to stays of up to 69 nights.<br>
            The majority of stays are relatively short, with a significant number of stays lasting 1 to 5 nights.<br>
            There's a steep decrease in the number of stays as the duration increases beyond 5 nights, indicating that longer stays are less common.<br>
            There are very few stays with durations exceeding 20 nights, suggesting that extended stays are relatively rare.<br><br>
            <h6 style='color: black; font-weight: bold; text-align: center;'>Stays in Weeknights</h6> 
            This dataset provides the count of stays based on the number of nights spent during weekdays (Monday to Friday).<br>
            The counts range from stays of 0 weeknights to stays of up to 50 weeknights.<br>
            The highest number of stays falls within the range of 1 to 3 weeknights, indicating that most guests spend a few nights during the weekdays.<br>
            There's a notable decrease in the number of stays as the duration in weeknights increases, with very few stays exceeding 10 weeknights.<br>
            Stays of 1 to 3 weeknights are the most common, suggesting that guests often stay for short periods during the weekdays, possibly for business or short trips.<br><br>
            <h6 style='color: black; font-weight: bold; text-align: center;'>Stays in Weekend Nights</h6> 
            This dataset provides the count of stays based on the number of nights spent during weekends (typically Friday and Saturday nights).<br>
            The counts range from stays of 0 weekend nights to stays of up to 19 weekend nights.<br>
            The majority of stays involve spending 1 or 2 nights during the weekend, indicating that weekend getaways are common.<br>
            There's a steep decrease in the number of stays as the duration in weekend nights increases beyond 2 nights, suggesting that longer weekend stays are less common.<br>
            Stays involving no weekend nights are relatively high, indicating that some guests may primarily stay during weekdays for business purposes.
        """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>{explanation}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        ####otherrr

        # Data
        average_stay_duration_data = {
            '0-30 days': 2.719427,
            '31-60 days': 3.423667,
            '61-90 days': 3.753439,
            '91-120 days': 3.857513,
            '121-150 days': 4.153005,
            'More than 150 days': 4.042293
        }

        average_weekend_nights_data = {
            '0-30 days': 0.743656,
            '31-60 days': 0.928445,
            '61-90 days': 1.053510,
            '91-120 days': 1.064148,
            '121-150 days': 1.167384,
            'More than 150 days': 1.058829
        }

        average_week_nights_data = {
            '0-30 days': 1.975771,
            '31-60 days': 2.495222,
            '61-90 days': 2.699928,
            '91-120 days': 2.793365,
            '121-150 days': 2.985621,
            'More than 150 days': 2.983464
        }

        header_html = """
            <div style="display: flex; justify-content: center;">
                <h1 style="text-align: center;">Average Stay Duration and Nights by Lead Time Interval</h1>
            </div>
        """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot Average Stay Duration
        plt.plot(average_stay_duration_data.keys(), average_stay_duration_data.values(), label='Average Stay Duration')

        # Plot Average Weekend Nights
        plt.plot(average_weekend_nights_data.keys(), average_weekend_nights_data.values(),
                 label='Average Weekend Nights')

        # Plot Average Week Nights
        plt.plot(average_week_nights_data.keys(), average_week_nights_data.values(), label='Average Week Nights')

        # Add labels and title
        plt.xlabel('Lead Time Interval')
        plt.ylabel('Average Value')
        plt.legend()

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Show the plot in Streamlit
        st.pyplot(plt)

        explanation = """
            <h6 style='color: black; font-weight: bold; text-align: center;'>Average Stay Duration by Lead Time Interval:</h6> 
            These numbers show the average total stay duration for different lead time intervals.<br>
            Lead time intervals range from 0-30 days to more than 150 days.<br>
            Generally, the average stay duration increases as the lead time interval increases.<br>
            Guests who book further in advance tend to stay longer, with the highest average stay duration observed for lead times of more than 150 days.<br>
            There's a gradual increase in average stay duration with each lead time interval, suggesting that guests who plan further ahead may be more likely to book longer stays.<br><br>
            <h6 style='color: black; font-weight: bold; text-align: center;'>Average Weekend Nights by Lead Time Interval:</h6> 
            These statistics show the average number of weekend nights (typically Friday and Saturday nights) spent based on different lead time intervals.<br>
            Similar to the average stay duration, the average number of weekend nights increases as the lead time interval increases.<br>
            Guests who book further in advance tend to spend more weekend nights at the hotel.<br>
            The increase in average weekend nights with longer lead times suggests that guests planning longer in advance may be more inclined to choose weekends for their stays, possibly for leisure or vacation purposes.<br><br>
            <h6 style='color: black; font-weight: bold; text-align: center;'>Average Weeknights by Lead Time Interval:</h6> 
            This dataset shows the average number of weeknights (typically Sunday to Thursday nights) spent based on different lead time intervals.<br>
            Like the previous datasets, there's an increasing trend in the average number of weeknights with longer lead time intervals.<br>
            Guests booking further in advance tend to stay more weeknights at the hotel.<br>
            The increase in average weeknights with longer lead times indicates that guests planning longer in advance may have more extended stays, possibly for business or leisure purposes.
        """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>{explanation}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='st-cz'>", unsafe_allow_html=True)

        #####otherr

        # Data for stays_in_weekend_nights
        cancellation_rate_weekend_nights = {
            0: 0.375470,
            1: 0.359530,
            2: 0.375068,
            3: 0.354633,
            4: 0.323227,
            5: 0.558442,
            6: 0.572368,
            7: 0.736842,
            8: 0.586207,
            9: 0.700000,
            10: 0.285714,
            12: 0.200000,
            13: 0.000000,
            14: 0.000000,
            16: 0.500000,
            18: 0.000000,
            19: 0.000000
        }

        # Data for stays_in_week_nights
        cancellation_rate_week_nights = {0: 0.250924, 1: 0.324508, 2: 0.441729, 3: 0.382222, 4: 0.365713, 5: 0.348844,
                                         6: 0.389558, 7: 0.315430, 8: 0.336391, 9: 0.416667, 10: 0.286408, 11: 0.618182,
                                         12: 0.547619, 13: 0.407407, 14: 0.885714, 15: 0.505882, 16: 0.733333,
                                         17: 0.500000, 18: 0.833333, 19: 0.674419, 20: 0.435897, 21: 0.733333,
                                         22: 1.000000, 24: 1.000000, 25: 0.000000, 26: 0.000000, 30: 0.250000,
                                         32: 0.000000, 33: 0.000000, 34: 0.000000, 40: 0.500000, 42: 0.000000,
                                         50: 0.000000}

        header_html = """
                    <div style="display: flex; justify-content: center;">
                        <h1 style="text-align: center;">Cancellation Rate by Stay Duration</h1>
                    </div>
                """

        # Display the centered header
        st.write(header_html, unsafe_allow_html=True)

        # Create a new figure
        plt.figure(figsize=(12, 6))

        # Plot for stays_in_weekend_nights
        plt.plot(cancellation_rate_weekend_nights.keys(), cancellation_rate_weekend_nights.values(),
                 label='Weekend Nights')

        # Plot for stays_in_week_nights
        plt.plot(cancellation_rate_week_nights.keys(), cancellation_rate_week_nights.values(), label='Week Nights')

        # Add labels and title
        plt.xlabel('Stay Duration')
        plt.ylabel('Cancellation Rate')

        plt.legend()

        # Show the plot in Streamlit
        st.pyplot(plt)

        explanation = """
            <h6 style='color: black; font-weight: bold; text-align: center;'>Cancellation Rate by Stay Duration (Weekend Nights):</h6>
            The cancellation rates based on the number of nights spent during weekends (typically Friday and Saturday nights).<br>
            Cancellation rates vary depending on the duration of weekend stays.<br>
            Generally, shorter weekend stays have lower cancellation rates, with stays of 1 to 4 nights having cancellation rates ranging from approximately 32.32% to 35.86%.<br>
            Longer weekend stays show higher cancellation rates, with stays of 5 nights or more having cancellation rates exceeding 55%.<br>
            There are exceptions, such as stays of 10 nights, which have a relatively low cancellation rate of approximately 28.57%.<br><br>
            <h6 style='color: black; font-weight: bold; text-align: center;'>Cancellation Rate by Stay Duration (Weeknights):</h6>
            The cancellation rates based on the number of nights spent during weekdays (typically Sunday to Thursday nights).<br>
            Cancellation rates vary depending on the duration of weeknight stays.<br>
            Similar to weekend stays, shorter weeknight stays generally have lower cancellation rates, with stays of 1 to 4 nights having cancellation rates ranging from approximately 31.57% to 38.22%.<br>
            Longer weeknight stays show higher cancellation rates, with stays of 5 nights or more having cancellation rates exceeding 50%.<br>
            There are exceptions, such as stays of 14 nights, which have an exceptionally high cancellation rate of approximately 88.57%.
        """

        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;'>{explanation}</div>",
            unsafe_allow_html=True
        )

    elif page == "Another Page":
        st.write("This is another page!")


# Additional explanations and analyses for other columns can be added here
if __name__ == '__main__':
    main()