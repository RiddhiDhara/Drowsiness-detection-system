import pandas as pd
from google import genai

CSV_PATH = "./python/drowsiness_data.csv"
# STAT_FILE = "stat.txt"


def summarize_data():
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key="AIzaSyBfISXWXfFtzEXBPtDSgq207ygCSP8gk5k")

        # Read the CSV file
        df = pd.read_csv(CSV_PATH)

        # Calculate alert level frequencies
        alert_counts = df['Alert Level'].value_counts()
        caution_count = alert_counts.get('Caution', 0)
        warning_count = alert_counts.get('Warning', 0)
        drowsiness_count = alert_counts.get('Drowsiness', 0)

        # Calculate sign frequencies
        blink_count = df['Signs Detected'].str.contains('Prolonged blink').sum()
        yawn_count = df['Signs Detected'].str.contains('Yawning').sum()
        head_tilt_count = df['Signs Detected'].str.contains('Head tilt').sum()

        # Prepare the prompt for Gemini API
        prompt = f"""
        Analyze the following drowsiness data for frequency and accident probability:
        Caution Alerts: {caution_count}
        Warning Alerts: {warning_count}
        Drowsiness Alerts: {drowsiness_count}
        Prolonged Blinks: {blink_count}
        Yawns: {yawn_count}
        Head Tilts: {head_tilt_count}
        Calculate the drowsiness frequency and the percentage chance of an accident.
        """

        # API Call to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        # Extract the result from API response
        summary = response.text

        # Write the summary to the stat.txt file
        with open("./python/stat.txt", "w") as file:
            file.write("Drowsiness Analysis Summary:\n")
            file.write(summary)

        print(summary)
        
        return summary

    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    summarize_data()
