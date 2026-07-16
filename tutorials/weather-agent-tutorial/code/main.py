from pathlib import Path

import requests

from nemantix.core import Agent, Expertise
from nemantix.core.tools import Toolset, tool
from nemantix.security import Verifier


class OpenMeteoTools(Toolset):
    @tool
    def get_weather_by_city(self, city_name: str) -> dict:
        """Fetches the weather for a city:
        Arguments:
         - city_name (str): the name of the city to query
        """
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city_name, "count": 1, "language": "it", "format": "json"}

        try:
            geo_response = requests.get(geocode_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                return {"status": "error", "error": f"City not found: '{city_name}'."}

            location = geo_data["results"][0]
            lat = location["latitude"]
            lon = location["longitude"]
            resolved_city = location["name"]

            print(f"[OpenMeteoTools] Found: {resolved_city} (Lat: {lat}, Lon: {lon})")

            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
            }

            weather_response = requests.get(weather_url, params=weather_params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            if "current_weather" in weather_data:
                current = weather_data["current_weather"]
                return {
                    "status": "success",
                    "city": resolved_city,
                    "temperature_celsius": current["temperature"],
                    "windspeed_kmh": current["windspeed"],
                    "is_day": current.get("is_day", 1),
                    "time": current["time"],
                }
            else:
                return {
                    "status": "error",
                    "error": "No data available for that locality.",
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"An error occurred during the communication with Open-Meteo: {str(e)}",
            }


def main() -> None:
    current_folder = Path.cwd()

    exp = Expertise.from_local_scripts(
        paths=[current_folder / "nxs/meteo-deliberate.nxs"],
        verifier=Verifier(current_folder / "keys/publickey.crt"),
    )

    agent = Agent(expertise=exp, build_on_start=True)

    while True:
        prompt = input("Insert the city name (or :exit to exit): ")

        if prompt == ":exit":
            break

        err, out = agent.run(user_request=f"Fetch the current weather of {prompt}")

        if err:
            print(f"An error occurred: {err}")
        else:
            print(f"Response: {out}")


if __name__ == "__main__":
    main()
