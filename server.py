from flask import Flask, request, render_template
import pickle
import numpy as np

# load the model
with open('bookings_cancellation_random_forest_model.pkl', 'rb') as file:
      model = pickle.load(file)


# # create a flask application
app = Flask(__name__)
#
#
@app.route("/", methods=["GET"])
def root():
    # read the file contents and send them to client
    return render_template('index.html')

ra3 = None
@app.route("/classify", methods=["POST"])
def classify():
    # get the values entered by user
    # print(request.form)

    no_of_adults = request.form.get("no_of_adults")
    no_of_children = request.form.get("no_of_children")
    no_of_weekend_nights = request.form.get("no_of_weekend_nights")
    no_of_week_nights = request.form.get("no_of_week_nights")

    meal_plans_array = ['Not Selected', 'Meal Plan 1', 'Meal Plan 2']
    inp = request.form.get("type_of_meal_plan")
    type_of_meal_plan = meal_plans_array.index(inp)

    required_car_parking_space = request.form.get("required_car_parking_space")

    room_types = ['room_type_1','room_type_2','room_type_3','room_type_4','room_type_5','room_type_6','room_type_7']
    ind = request.form.get("room_type_reserved")
    room_type_reserved = room_types.index(ind)

    repeated_guest = request.form.get("repeated_guest")
    no_of_previous_cancellations = request.form.get("no_of_previous_cancellations")
    no_of_previous_bookings_not_canceled = request.form.get("no_of_previous_bookings_not_canceled")
    no_of_special_requests = request.form.get("no_of_special_requests")

    market_segments = ['aviation','complementary','corporate','online','offline']
    market_segments_ohe = np.zeros(len(market_segments))
    inp2 = request.form.get("market_segment")
    market_segments_ohe[market_segments.index(inp2)] = 1

    lead_time_scaled = request.form.get("lead_time_scaled")
    avg_price_per_room_scaled = request.form.get("avg_price_per_room_scaled")

    answers = np.array([[no_of_adults,no_of_children,no_of_weekend_nights ,no_of_week_nights,type_of_meal_plan,required_car_parking_space, room_type_reserved, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, no_of_special_requests, lead_time_scaled, avg_price_per_room_scaled]])

    market_segments_ohe = market_segments_ohe.reshape(1,-1)

    # result = np.concatenate((array_2d[:, :4], array_1d_reshaped, array_2d[:, 4:]), axis=1)

    answers2 = np.concatenate((answers[:, :10], market_segments_ohe, answers[:,10:]), axis=1)
    print(answers2.shape)

    if model.predict(answers2) == 1:
         return "Not cancelled...."
    else:
        return "Will be CANCELLED...."


# start the application
app.run(host="0.0.0.0", port=8000, debug=True)
