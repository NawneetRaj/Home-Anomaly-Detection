def check_alert(person_count, threshold=50):
    if person_count > threshold:
        return True
    return False
