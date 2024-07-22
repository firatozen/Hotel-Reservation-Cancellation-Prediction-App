
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

#################
def main(customer_object_id=None):
    page = st.experimental_get_query_params().get("page", [None])[0]

    # Check if the page parameter is missing or invalid
    if page not in ["Database (Predict)", "Simulation", "Insights", "Reservation", "Optimize Revenue"]:
        # Default to the database page
        page = "Database(Predict)"

    # Display the sidebar for page selection
    selected_page = st.sidebar.selectbox("Select Page",
                                         ["Database (Predict)", "Simulation", "Insights", "Reservation", "Optimize Revenue"])

    # Update the page parameter based on the selected page
    if selected_page == "Database (Predict)":
        page = "Database (Predict)"
    elif selected_page == "Simulation":
        page = "Simulation"
    elif selected_page == "Insights":
        page = "Insights"
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
    elif page == "Reservation":
        # Display the reservation page
        with st.expander("Reservation"):
            reservation_page()
    elif page == "Optimize Revenue":
        # Display the reservation page
        with st.expander("Optimize Revenue"):
            optimize_revenue()


# Additional explanations and analyses for other columns can be added here
if __name__ == '__main__':
    main()
