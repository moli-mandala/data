from geopy.geocoders import Nominatim

def geolocate(place):
    geolocator = Nominatim(user_agent="Aryaman Arora")
    location = geolocator.geocode(place)
    if location is not None:
        print(location)
        print(f"{location.latitude},{location.longitude}")
    else:
        print("No location found")

while True:
    place = input("Enter a place: ")
    geolocate(place)
