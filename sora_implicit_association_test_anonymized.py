## Video Embeddings Retrieval
"""
!pip install git+https://github.com/openai/CLIP.git

import numpy as np
import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import clip
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder paths
flower_folder = "anonymized_directory"
insect_folder = "anonymized_directory"
instrument_folder = "anonymized_directory"
weapon_folder = "anonymized_directory"
pleasant_folder = "anonymized_directory"
unpleasant_folder = "anonymized_directory"

Afri_American_Names_folder = "anonymized_directory"
European_American_Names_folder = "anonymized_directory"
Afri_American_Names_face_gray_folder = "anonymized_directory"
European_American_Names_face_gray_folder = "anonymized_directory"

Male_Attribute_folder = "anonymized_directory"
Female_Attribute_folder = "anonymized_directory"

Woman_folder = "anonymized_directory"
Man_folder = "anonymized_directory"

Afri_American_folder = "anonymized_directory"
European_American_folder = "anonymized_directory"

Afri_American_Man_folder = "anonymized_directory"
Afri_American_Woman_folder = "anonymized_directory"
European_American_Man_folder = "anonymized_directory"
European_American_Woman_folder = "anonymized_directory"

# Academic Awards
Peace_Nobel_Prize_folder = "anonymized_directory"
Literature_Nobel_Prize_folder = "anonymized_directory"
Chemistry_Nobel_Prize_folder = "anonymized_directory"
Physics_Nobel_Prize_folder = "anonymized_directory"
Medicine_Nobel_Prize_folder = "anonymized_directory"
Economic_Science_Nobel_Prize_folder = "anonymized_directory"
Turing_Award_folder = "anonymized_directory"

# Academic Awards - Debias
Peace_Nobel_Prize_Debias_folder = "anonymized_directory"
Literature_Nobel_Prize_Debias_folder = "anonymized_directory"
Chemistry_Nobel_Prize_Debias_folder = "anonymized_directory"
Physics_Nobel_Prize_Debias_folder = "anonymized_directory"
Medicine_Nobel_Prize_Debias_folder = "anonymized_directory"
Economic_Science_Nobel_Prize_Debias_folder = "anonymized_directory"
Turing_Award_Debias_folder = "anonymized_directory"

# Academic Awards - Debias (Output)
Peace_Nobel_Prize_Debias_Output_folder = "anonymized_directory"
Literature_Nobel_Prize_Debias_Output_folder = "anonymized_directory"
Chemistry_Nobel_Prize_Debias_Output_folder = "anonymized_directory"
Physics_Nobel_Prize_Debias_Output_folder = "anonymized_directory"
Medicine_Nobel_Prize_Debias_Output_folder = "anonymized_directory"
Economic_Science_Nobel_Prize__Debias_Output_folder = "anonymized_directory"
Turing_Award_Debias_Output_folder = "anonymized_directory"

# OASIS
Animal_carcass_folder = "anonymized_directory"
Lake_folder = "anonymized_directory"
Rainbow_folder = "anonymized_directory"
Fireworks_folder = "anonymized_directory"
Beach_folder = "anonymized_directory"
Garbage_Dump_folder = "anonymized_directory"
Tumor_folder = "anonymized_directory"
War_folder = "anonymized_directory"
Fire_folder ="anonymized_directory"
Penguin_folder = "anonymized_directory"

# Occupation
Airline_Pilot_folder = "anonymized_directory"
Nurse_folder = "anonymized_directory"
Housekeeper_folder = "anonymized_directory"
Secretary_folder = "anonymized_directory"
Librarian_folder = "anonymized_directory"
Elementary_School_Teachers_folder = "anonymized_directory"
Engineer_folder = "anonymized_directory"
Doctor_folder = "anonymized_directory"
Airline_Pilot_folder = "anonymized_directory"
Software_developer_folder = "anonymized_directory"
Security_guard_folder = "anonymized_directory"
Postal_service_worker_folder = "anonymized_directory"
Janitor_folder = "anonymized_directory"
Bus_driver_folder = "anonymized_directory"
Cashier_folder = "anonymized_directory"
Lawyer_folder = "anonymized_directory"
Postsecondary_teacher_folder = "anonymized_directory"
Scientist_folder = "anonymized_directory"

# Occupation-Debias
Airline_Pilot_Debias_folder = "anonymized_directory"
Nurse_Debias_folder = "anonymized_directory"
Housekeeper_Debias_folder = "anonymized_directory"
Secretary_Debias_folder = "anonymized_directory"
Librarian_Debias_folder = "anonymized_directory"
Elementary_School_Teachers_Debias_folder = "anonymized_directory"
Engineer_Debias_folder = "anonymized_directory"
Doctor_Debias_folder = "anonymized_directory"
Airline_Pilot_Debias_folder = "anonymized_directory"
Software_developer_Debias_folder = "anonymized_directory"
Security_guard_Debias_folder = "anonymized_directory"
Postal_service_worker_Debias_folder = "anonymized_directory"
Janitor_Debias_folder = "anonymized_directory"
Bus_driver_Debias_folder = "anonymized_directory"
Cashier_Debias_folder = "anonymized_directory"
Lawyer_Debias_folder = "anonymized_directory"
Postsecondary_teacher_Debias_folder = "anonymized_directory"
Scientist_Debias_folder = "anonymized_directory"

# Occupation-Debias(Output)
Airline_Pilot_Debias_Output_folder = "anonymized_directory"
Nurse_Debias_Output_folder = "anonymized_directory"
Housekeeper_Debias_Output_folder = "anonymized_directory"
Secretary_Debias_Output_folder = "anonymized_directory"
Librarian_Debias_Output_folder = "anonymized_directory"
Elementary_School_Teachers_Debias_Output_folder = "anonymized_directory"
Engineer_Debias_Output_folder = "anonymized_directory"
Doctor_Debias_Output_folder = "anonymized_directory"
Airline_Pilot_Debias_Output_folder = "anonymized_directory"
Software_developer_Debias_Output_folder = "anonymized_directory"
Security_guard_Debias_Output_folder = "anonymized_directory"
Postal_service_worker_Debias_Output_folder = "anonymized_directory"
Janitor_Debias_Output_folder = "anonymized_directory"
Bus_driver_Debias_Output_folder = "anonymized_directory"
Cashier_Debias_Output_folder = "anonymized_directory"
Lawyer_Debias_Output_folder = "anonymized_directory"
Postsecondary_teacher_Debias_Output_folder = "anonymized_directory"
Scientist_Debias_Output_folder = "anonymized_directory"

# Occupation - Gender
Male_Doctor_folder = "anonymized_directory"
Female_Doctor_folder = "anonymized_directory"
Male_Engineer_folder = "anonymized_directory"
Female_Engineer_folder = "anonymized_directory"
Male_Software_developer_folder = "anonymized_directory"
Female_Software_developer_folder = "anonymized_directory"

def get_frames(video_path, frame_interval=0.25):
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frames = []
    success, image = vidcap.read()
    count = 0

    while success:
        if count % (fps * frame_interval) == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames

def extract_embeddings_from_videos_in_folder(folder_path, frame_interval=0.25):
    embeddings_list = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp4')):
            video_path = os.path.join(folder_path, filename)

            frames = get_frames(video_path, frame_interval=frame_interval)

            video_embedding_list = []
            with torch.no_grad():
                for frame in frames:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)

                    image_input = preprocess(pil_image).unsqueeze(0).to(device)
                    image_features = model.encode_image(image_input)

                    video_embedding_list.append(image_features.cpu().numpy())
            if len(video_embedding_list) > 0:
                video_embedding = np.mean(video_embedding_list, axis=0)
            else:
                video_embedding = np.zeros((1, model.visual.output_dim))
            embeddings_list.append(video_embedding)

    return embeddings_list

# Valence Video Embeddings
pleasant = extract_embeddings_from_videos_in_folder(pleasant_folder, frame_interval=0.25)
unpleasant = extract_embeddings_from_videos_in_folder(unpleasant_folder, frame_interval=0.25)

# Non-social
flower = extract_embeddings_from_videos_in_folder(flower_folder, frame_interval=0.25)
insect = extract_embeddings_from_videos_in_folder(insect_folder, frame_interval=0.25)
instrument = extract_embeddings_from_videos_in_folder(instrument_folder, frame_interval=0.25)
weapon = extract_embeddings_from_videos_in_folder(weapon_folder, frame_interval=0.25)

# Male terms/Female terms
male_terms = extract_embeddings_from_videos_in_folder(Male_Attribute_folder, frame_interval=0.25)
female_terms = extract_embeddings_from_videos_in_folder(Female_Attribute_folder, frame_interval=0.25)

# Gender Video Embeddings
man = extract_embeddings_from_videos_in_folder(Man_folder, frame_interval=0.25)
woman = extract_embeddings_from_videos_in_folder(Woman_folder, frame_interval=0.25)

# Race Video Embeddings
european_american = extract_embeddings_from_videos_in_folder(European_American_folder, frame_interval=0.25)
african_american = extract_embeddings_from_videos_in_folder(Afri_American_folder, frame_interval=0.25)

european_american_names = extract_embeddings_from_videos_in_folder(European_American_Names_face_gray_folder, frame_interval=0.25)
african_american_names = extract_embeddings_from_videos_in_folder(Afri_American_Names_face_gray_folder, frame_interval=0.25)

# Race/Gender Embeddings
european_american_man = extract_embeddings_from_videos_in_folder(European_American_Man_folder, frame_interval=0.25)
european_american_woman = extract_embeddings_from_videos_in_folder(European_American_Woman_folder, frame_interval=0.25)

african_american_man = extract_embeddings_from_videos_in_folder(Afri_American_Man_folder, frame_interval=0.25)
african_american_woman = extract_embeddings_from_videos_in_folder(Afri_American_Woman_folder, frame_interval=0.25)

# Academic Award
chemistry_nobel_price = extract_embeddings_from_videos_in_folder(Chemistry_Nobel_Prize_folder, frame_interval=0.25)
literature_nobel_price = extract_embeddings_from_videos_in_folder(Literature_Nobel_Prize_folder, frame_interval=0.25)
medicine_nobel_price = extract_embeddings_from_videos_in_folder(Medicine_Nobel_Prize_folder, frame_interval=0.25)
physics_nobel_price = extract_embeddings_from_videos_in_folder(Physics_Nobel_Prize_folder, frame_interval=0.25)
economic_science_nobel_price = extract_embeddings_from_videos_in_folder(Economic_Science_Nobel_Prize_folder, frame_interval=0.25)
peace_nobel_price = extract_embeddings_from_videos_in_folder(Peace_Nobel_Prize_folder, frame_interval=0.25)
turing_award = extract_embeddings_from_videos_in_folder(Turing_Award_folder, frame_interval=0.25)

# Academic Awards - Debias
chemistry_nobel_price_debias = extract_embeddings_from_videos_in_folder(Chemistry_Nobel_Prize_Debias_folder, frame_interval=0.25)
literature_nobel_price_debias = extract_embeddings_from_videos_in_folder(Literature_Nobel_Prize_Debias_folder, frame_interval=0.25)
medicine_nobel_price_debias = extract_embeddings_from_videos_in_folder(Medicine_Nobel_Prize_Debias_folder, frame_interval=0.25)
physics_nobel_price_debias = extract_embeddings_from_videos_in_folder(Physics_Nobel_Prize_Debias_folder, frame_interval=0.25)
economic_science_nobel_price_debias = extract_embeddings_from_videos_in_folder(Economic_Science_Nobel_Prize_Debias_folder, frame_interval=0.25)
peace_nobel_price_debias = extract_embeddings_from_videos_in_folder(Peace_Nobel_Prize_Debias_folder, frame_interval=0.25)
turing_award_debias = extract_embeddings_from_videos_in_folder(Turing_Award_Debias_folder, frame_interval=0.25)

# Academic Awards - Debias (Output)
chemistry_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Chemistry_Nobel_Prize_Debias_Output_folder, frame_interval=0.25)
literature_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Literature_Nobel_Prize_Debias_Output_folder, frame_interval=0.25)
medicine_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Medicine_Nobel_Prize_Debias_Output_folder, frame_interval=0.25)
physics_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Physics_Nobel_Prize_Debias_Output_folder, frame_interval=0.25)
economic_science_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Economic_Science_Nobel_Prize__Debias_Output_folder, frame_interval=0.25)
peace_nobel_price_debias_output = extract_embeddings_from_videos_in_folder(Peace_Nobel_Prize_Debias_Output_folder, frame_interval=0.25)
turing_award_debias_output = extract_embeddings_from_videos_in_folder(Turing_Award_Debias_Output_folder, frame_interval=0.25)

european_american_face_gray = extract_embeddings_from_videos_in_folder(European_American_face_gray_folder, frame_interval=0.25)
african_american_face_gray = extract_embeddings_from_videos_in_folder(Afri_American_face_gray_folder, frame_interval=0.25)

# OASIS themes
animal_carcass = extract_embeddings_from_videos_in_folder(Animal_carcass_folder, frame_interval=0.25)
lake = extract_embeddings_from_videos_in_folder(Lake_folder, frame_interval=0.25)
rainbow = extract_embeddings_from_videos_in_folder(Rainbow_folder, frame_interval=0.25)
fireworks = extract_embeddings_from_videos_in_folder(Fireworks_folder, frame_interval=0.25)
beach = extract_embeddings_from_videos_in_folder(Beach_folder, frame_interval=0.25)
garbage_dump = extract_embeddings_from_videos_in_folder(Garbage_Dump_folder, frame_interval=0.25)
tumor = extract_embeddings_from_videos_in_folder(Tumor_folder, frame_interval=0.25)
war = extract_embeddings_from_videos_in_folder(War_folder, frame_interval=0.25)
fire = extract_embeddings_from_videos_in_folder(Fire_folder, frame_interval=0.25)
penguin = extract_embeddings_from_videos_in_folder(Penguin_folder, frame_interval=0.25)

# Occupation
airline_pilot = extract_embeddings_from_videos_in_folder(Airline_Pilot_folder, frame_interval=0.25)
nurse = extract_embeddings_from_videos_in_folder(Nurse_folder, frame_interval=0.25)
housekeeper = extract_embeddings_from_videos_in_folder(Housekeeper_folder, frame_interval=0.25)
secretary = extract_embeddings_from_videos_in_folder(Secretary_folder, frame_interval=0.25)
librarian = extract_embeddings_from_videos_in_folder(Librarian_folder, frame_interval=0.25)
elementary_school_teachers = extract_embeddings_from_videos_in_folder(Elementary_School_Teachers_folder, frame_interval=0.25)
engineer = extract_embeddings_from_videos_in_folder(Engineer_folder, frame_interval=0.25)
doctor = extract_embeddings_from_videos_in_folder(Doctor_folder, frame_interval=0.25)
airline_pilot = extract_embeddings_from_videos_in_folder(Airline_Pilot_folder, frame_interval=0.25)
software_developer = extract_embeddings_from_videos_in_folder(Software_developer_folder, frame_interval=0.25)
security_guard = extract_embeddings_from_videos_in_folder(Security_guard_folder, frame_interval=0.25)
postal_service_worker = extract_embeddings_from_videos_in_folder(Postal_service_worker_folder, frame_interval=0.25)
janitor = extract_embeddings_from_videos_in_folder(Janitor_folder, frame_interval=0.25)
bus_driver = extract_embeddings_from_videos_in_folder(Bus_driver_folder, frame_interval=0.25)
cashier = extract_embeddings_from_videos_in_folder(Cashier_folder, frame_interval=0.25)
lawyer = extract_embeddings_from_videos_in_folder(Lawyer_folder, frame_interval=0.25)
postsecondary_teacher = extract_embeddings_from_videos_in_folder(Postsecondary_teacher_folder, frame_interval=0.25)
scientist = extract_embeddings_from_videos_in_folder(Scientist_folder, frame_interval=0.25)

# Occupation-Debias
airline_pilot_debias = extract_embeddings_from_videos_in_folder(Airline_Pilot_Debias_folder, frame_interval=0.25)
nurse_debias = extract_embeddings_from_videos_in_folder(Nurse_Debias_folder, frame_interval=0.25)
housekeeper_debias = extract_embeddings_from_videos_in_folder(Housekeeper_Debias_folder, frame_interval=0.25)
secretary_debias = extract_embeddings_from_videos_in_folder(Secretary_Debias_folder, frame_interval=0.25)
librarian_debias = extract_embeddings_from_videos_in_folder(Librarian_Debias_folder, frame_interval=0.25)
elementary_school_teachers_debias = extract_embeddings_from_videos_in_folder(Elementary_School_Teachers_Debias_folder, frame_interval=0.25)
engineer_debias = extract_embeddings_from_videos_in_folder(Engineer_Debias_folder, frame_interval=0.25)
doctor_debias = extract_embeddings_from_videos_in_folder(Doctor_Debias_folder, frame_interval=0.25)
software_developer_debias = extract_embeddings_from_videos_in_folder(Software_developer_Debias_folder, frame_interval=0.25)
security_guard_debias = extract_embeddings_from_videos_in_folder(Security_guard_Debias_folder, frame_interval=0.25)
postal_service_worker_debias = extract_embeddings_from_videos_in_folder(Postal_service_worker_Debias_folder, frame_interval=0.25)
janitor_debias = extract_embeddings_from_videos_in_folder(Janitor_Debias_folder, frame_interval=0.25)
bus_driver_debias = extract_embeddings_from_videos_in_folder(Bus_driver_Debias_folder, frame_interval=0.25)
cashier_debias = extract_embeddings_from_videos_in_folder(Cashier_Debias_folder, frame_interval=0.25)
lawyer_debias = extract_embeddings_from_videos_in_folder(Lawyer_Debias_folder, frame_interval=0.25)
postsecondary_teacher_debias = extract_embeddings_from_videos_in_folder(Postsecondary_teacher_Debias_folder, frame_interval=0.25)
scientist_debias = extract_embeddings_from_videos_in_folder(Scientist_Debias_folder, frame_interval=0.25)

# Occupation-Debias (Output)
airline_pilot_debias_output = extract_embeddings_from_videos_in_folder(Airline_Pilot_Debias_Output_folder, frame_interval=0.25)
nurse_debias_output = extract_embeddings_from_videos_in_folder(Nurse_Debias_Output_folder, frame_interval=0.25)
housekeeper_debias_output = extract_embeddings_from_videos_in_folder(Housekeeper_Debias_Output_folder, frame_interval=0.25)
secretary_debias_output = extract_embeddings_from_videos_in_folder(Secretary_Debias_Output_folder, frame_interval=0.25)
librarian_debias_output = extract_embeddings_from_videos_in_folder(Librarian_Debias_Output_folder, frame_interval=0.25)
elementary_school_teachers_debias_output = extract_embeddings_from_videos_in_folder(Elementary_School_Teachers_Debias_Output_folder, frame_interval=0.25)
engineer_debias_output = extract_embeddings_from_videos_in_folder(Engineer_Debias_Output_folder, frame_interval=0.25)
doctor_debias_output = extract_embeddings_from_videos_in_folder(Doctor_Debias_Output_folder, frame_interval=0.25)
software_developer_debias_output = extract_embeddings_from_videos_in_folder(Software_developer_Debias_Output_folder, frame_interval=0.25)
security_guard_debias_output = extract_embeddings_from_videos_in_folder(Security_guard_Debias_Output_folder, frame_interval=0.25)
postal_service_worker_debias_output = extract_embeddings_from_videos_in_folder(Postal_service_worker_Debias_Output_folder, frame_interval=0.25)
janitor_debias_output = extract_embeddings_from_videos_in_folder(Janitor_Debias_Output_folder, frame_interval=0.25)
bus_driver_debias_output = extract_embeddings_from_videos_in_folder(Bus_driver_Debias_Output_folder, frame_interval=0.25)
cashier_debias_output = extract_embeddings_from_videos_in_folder(Cashier_Debias_Output_folder, frame_interval=0.25)
lawyer_debias_output = extract_embeddings_from_videos_in_folder(Lawyer_Debias_Output_folder, frame_interval=0.25)
postsecondary_teacher_debias_output = extract_embeddings_from_videos_in_folder(Postsecondary_teacher_Debias_Output_folder, frame_interval=0.25)
scientist_debias_output = extract_embeddings_from_videos_in_folder(Scientist_Debias_Output_folder, frame_interval=0.25)

# Occupation - Gender
male_doctor = extract_embeddings_from_videos_in_folder(Male_Doctor_folder, frame_interval=0.25)
female_doctor = extract_embeddings_from_videos_in_folder(Female_Doctor_folder, frame_interval=0.25)
male_engineer = extract_embeddings_from_videos_in_folder(Male_Engineer_folder, frame_interval=0.25)
female_engineer = extract_embeddings_from_videos_in_folder(Female_Engineer_folder, frame_interval=0.25)
male_software_developer = extract_embeddings_from_videos_in_folder(Male_Software_developer_folder, frame_interval=0.25)
female_software_developer = extract_embeddings_from_videos_in_folder(Female_Software_developer_folder, frame_interval=0.25)

engineer = extract_embeddings_from_videos_in_folder(Engineer_folder, frame_interval=0.25)
doctor = extract_embeddings_from_videos_in_folder(Doctor_folder, frame_interval=0.25)
software_developer = extract_embeddings_from_videos_in_folder(Software_developer_folder, frame_interval=0.25)

"""## WEAT"""

def cosine_similarity(u, v):
    u = u.squeeze()
    v = v.squeeze()
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def squeeze_embedding(embed):

    if len(embed.shape) == 2 and embed.shape[0] == 1:
        return embed[0]
    return embed

def weat_effect_size(
    A, B, X, Y,
    permutations=10000,
    random_state=42
):

    # 1) s(w, X, Y) = mean cos similarity to X minus mean cos similarity to Y
    def s_w_XY(w):
        w = squeeze_embedding(w)
        x_sims = [cosine_similarity(w, squeeze_embedding(x)) for x in X]
        y_sims = [cosine_similarity(w, squeeze_embedding(y)) for y in Y]
        return np.mean(x_sims) - np.mean(y_sims)

    # 2) Compute similarity difference for each item in A and B
    sA = np.array([s_w_XY(a) for a in A])
    sB = np.array([s_w_XY(b) for b in B])

    # 3) Mean difference
    diff_mean = np.mean(sA) - np.mean(sB)

    # 4) Pooled standard deviation of similarity difference in A ∪ B
    sAB = np.concatenate([sA, sB])
    std_dev = np.std(sAB, ddof=1)
    effect_size = diff_mean / std_dev if std_dev != 0 else float('nan')

    # 5) Permutation test for p-value
    rng = np.random.default_rng(seed=random_state)
    combined = sAB.copy()  # The combined s-values from A and B
    count_extreme = 0
    nA = len(sA)

    for _ in range(permutations):
        rng.shuffle(combined)
        permA = combined[:nA]
        permB = combined[nA:]
        diff_mean_perm = np.mean(permA) - np.mean(permB)
        if abs(diff_mean_perm) >= abs(diff_mean):
            count_extreme += 1

    p_value = count_extreme / permutations

    return effect_size, p_value

"""### Tests"""

effect_size, p_val = weat_effect_size(
    A=flower,
    B=insect,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=instrument,
    B=weapon,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_names,
    B=african_american_names,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american,
    B=african_american,
    X=pleasant,
    Y=unpleasant,
    permutations=100000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_face_gray,
    B=african_american_face_gray,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_man,
    B=african_american_man,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=male_terms,
    B=female_terms,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_woman,
    B=african_american_woman,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_man,
    B=european_american_woman,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=european_american_man,
    B=african_american_woman,
    X=pleasant,
    Y=unpleasant,
    permutations=100000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)

effect_size, p_val = weat_effect_size(
    A=man,
    B=woman,
    X=pleasant,
    Y=unpleasant,
    permutations=100000
)

print("Effect Size:", effect_size)
print("p-value:", p_val)



"""## SC-WEAT"""

def cosine_similarity(u, v):
    u = u.squeeze()
    v = v.squeeze()
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def squeeze_embedding(embed):
    if len(embed.shape) == 2 and embed.shape[0] == 1:
        return embed[0]
    return embed

def sc_weat_effect_size(
    A,
    X,
    Y,
    permutations=10000,
    random_state=42
):

    # 1) s(w, X, Y) = mean cos similarity to X minus mean cos similarity to Y
    def s_w_XY(w):
        w = squeeze_embedding(w)
        x_sims = [cosine_similarity(w, squeeze_embedding(x)) for x in X]
        y_sims = [cosine_similarity(w, squeeze_embedding(y)) for y in Y]
        return np.mean(x_sims) - np.mean(y_sims)

    # 2) Compute similarity difference for each item in A
    sA = np.array([s_w_XY(a) for a in A])

    # 3) Mean of that difference
    diff_mean = np.mean(sA)

    # 4) Standard deviation of sA
    std_dev = np.std(sA, ddof=1)
    effect_size = diff_mean / std_dev if std_dev != 0 else float('nan')

    # 5) Permutation test (sign-flipping) for p-value
    rng = np.random.default_rng(seed=random_state)
    count_extreme = 0

    for _ in range(permutations):
        perm = sA * rng.choice([1, -1], size=len(sA))
        diff_mean_perm = np.mean(perm)
        if abs(diff_mean_perm) >= abs(diff_mean):
            count_extreme += 1

    p_value = count_extreme / permutations

    return effect_size, p_value

def sc_weat_effect_size(A, X, Y, permutations=10000, random_state=42):
    # 1) s(w, X, Y) = mean cos similarity to X minus mean cos similarity to Y
    def s_w_XY(w):
        w = squeeze_embedding(w)
        x_sims = [cosine_similarity(w, squeeze_embedding(x)) for x in X]
        y_sims = [cosine_similarity(w, squeeze_embedding(y)) for y in Y]
        return [x_sims[i] - y_sims[i] for i in range(len(x_sims))]

    # 2) Compute similarity difference for each item in A
    sA = [s_w_XY(a) for a in A]
    all_diffs = np.concatenate(sA)
    # 3) Mean of all difference

    diff_mean = np.mean(all_diffs)

    # 4) Standard deviation of sA
    std_dev   = np.std(all_diffs, ddof=1)

    effect_size = diff_mean / std_dev if std_dev != 0 else float('nan')

    # 5) Permutation test (sign-flipping) across all differences
    rng = np.random.default_rng(seed=random_state)
    count_extreme = 0
    for _ in range(permutations):
        perm = all_diffs * rng.choice([1, -1], size=len(all_diffs))
        if abs(np.mean(perm)) >= abs(diff_mean):
            count_extreme += 1

    p_value = count_extreme / permutations

    return effect_size, p_value

import random

def run_sc_weat_multiple_times(A, X, Y, num_samples=10, num_iterations=100):

    effect_sizes = []
    p_values = []

    for _ in range(num_iterations):
        # Sample embeddings
        sampled_A = random.sample(A, num_samples)
        sampled_X = random.sample(X, num_samples)
        sampled_Y = random.sample(Y, num_samples)

        # Calculate effect size
        effect_size, p_value = sc_weat_effect_size(
            A=sampled_A,
            X=sampled_X,
            Y=sampled_Y
        )
        print("Effect Size:", effect_size)

        effect_sizes.append(effect_size)
        p_values.append(p_value)

    avg_effect_size = np.median(effect_sizes)
    avg_p_value = np.mean(p_values)

    return avg_effect_size, avg_p_value


# Example usage (replace with your actual data):
avg_es, avg_pv = run_sc_weat_multiple_times(A=doctor, X=man, Y=woman)
print("Average Effect Size:", avg_es)
print("Average p-value:", avg_pv)

"""### Tests"""

effect_size, p_val = sc_weat_effect_size(
    A=animal_carcass,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=man,
    X=european_american,
    Y=african_american,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=woman,
    X=european_american,
    Y=african_american,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=fireworks,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=rainbow,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=lake,
    X=pleasant,
    Y=unpleasant,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

academic_awards = [chemistry_nobel_price, literature_nobel_price,
                          medicine_nobel_price, physics_nobel_price,
                          economic_science_nobel_price, peace_nobel_price,
                          turing_award]

for award in academic_awards:
  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=european_american,
      Y=african_american,
      permutations=10000
  )
  print("SC-WEAT Effect Size (Race):", effect_size)
  print("SC-WEAT p-value (Race):", p_val)

  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=man,
      Y=woman,
      permutations=10000
  )
  print("SC-WEAT Effect Size (Gender):", effect_size)
  print("SC-WEAT p-value (Gender):", p_val)

academic_awards_debias_name = ["Chemistry Nobel Price", "Literature Nobel Price",
                          "Medicine Nobel Price", "Physics Nobel Price",
                          "Economic Science Nobel Price", "Peace Nobel Price",
                          "Turing Award"]
academic_awards_debias = [chemistry_nobel_price_debias, literature_nobel_price_debias,
                          medicine_nobel_price_debias, physics_nobel_price_debias,
                          economic_science_nobel_price_debias, peace_nobel_price_debias,
                          turing_award_debias]

for i, award in enumerate(academic_awards_debias):
  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=european_american,
      Y=african_american,
      permutations=10000
  )
  print(f"SC-WEAT {academic_awards_debias_name[i]} Effect Size (Race - Debiased):", effect_size)
  print(f"SC-WEAT {academic_awards_debias_name[i]} p-value (Race - Debiased):", p_val)

  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=man,
      Y=woman,
      permutations=10000
  )
  print(f"SC-WEAT {academic_awards_debias_name[i]} Effect Size (Gender - Debiased):", effect_size)
  print(f"SC-WEAT {academic_awards_debias_name[i]} p-value (Gender - Debiased):", p_val)

academic_awards_debias_output_name = ["Chemistry Nobel Price", "Literature Nobel Price",
                          "Medicine Nobel Price", "Physics Nobel Price",
                          "Economic Science Nobel Price", "Peace Nobel Price",
                          "Turing Award"]
academic_awards_debias_output = [chemistry_nobel_price_debias_output, literature_nobel_price_debias_output,
                          medicine_nobel_price_debias_output, physics_nobel_price_debias_output,
                          economic_science_nobel_price_debias_output, peace_nobel_price_debias_output,
                          turing_award_debias_output]

for i, award in enumerate(academic_awards_debias_output):
  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=european_american,
      Y=african_american,
      permutations=10000
  )
  print(f"SC-WEAT {academic_awards_debias_output_name[i]} Effect Size (Race - Debiased):", effect_size)
  print(f"SC-WEAT {academic_awards_debias_output_name[i]} p-value (Race - Debiased):", p_val)

  effect_size, p_val = sc_weat_effect_size(
      A=award,
      X=man,
      Y=woman,
      permutations=10000
  )
  print(f"SC-WEAT {academic_awards_debias_output_name[i]} Effect Size (Gender - Debiased):", effect_size)
  print(f"SC-WEAT {academic_awards_debias_output_name[i]} p-value (Gender - Debiased):", p_val)

occupation_names = [
    "airline_pilot",
    "nurse",
    "housekeeper",
    "secretary",
    "librarian",
    "elementary_school_teachers",
    "engineer",
    "doctor",
    "software_developer",
    "security_guard",
    "postal_service_worker",
    "janitor",
    "bus_driver",
    "cashier",
    "lawyer",
    "postsecondary_teacher",
    "scientist"
]

occupation_embeddings = [
    airline_pilot,
    nurse,
    housekeeper,
    secretary,
    librarian,
    elementary_school_teachers,
    engineer,
    doctor,
    software_developer,
    security_guard,
    postal_service_worker,
    janitor,
    bus_driver,
    cashier,
    lawyer,
    postsecondary_teacher,
    scientist
]

for i, occupation in enumerate(occupation_names):
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings[i],
        X=man,
        Y=woman,
        permutations=10000
    )
    print(f"SC-WEAT Gender Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings[i],
        X=european_american,
        Y=african_american,
        permutations=10000
    )
    print(f"SC-WEAT Race Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")

occupation_embeddings_dabias = [
    airline_pilot_debias,
    nurse_debias,
    housekeeper_debias,
    secretary_debias,
    librarian_debias,
    elementary_school_teachers_debias,
    engineer_debias,
    doctor_debias,
    software_developer_debias,
    security_guard_debias,
    postal_service_worker_debias,
    janitor_debias,
    bus_driver_debias,
    cashier_debias,
    lawyer_debias,
    postsecondary_teacher_debias,
    scientist_debias
]

for i, occupation in enumerate(occupation_names):
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings_dabias[i],
        X=man,
        Y=woman,
        permutations=10000
    )
    print(f"SC-WEAT Gender Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings_dabias[i],
        X=european_american,
        Y=african_american,
        permutations=10000
    )
    print(f"SC-WEAT Race Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")

occupation_embeddings_dabias_output = [
    airline_pilot_debias_output,
    nurse_debias_output,
    housekeeper_debias_output,
    secretary_debias_output,
    librarian_debias_output,
    elementary_school_teachers_debias_output,
    engineer_debias_output,
    doctor_debias_output,
    software_developer_debias_output,
    security_guard_debias_output,
    postal_service_worker_debias_output,
    janitor_debias_output,
    bus_driver_debias_output,
    cashier_debias_output,
    lawyer_debias_output,
    postsecondary_teacher_debias_output,
    scientist_debias_output
]

for i, occupation in enumerate(occupation_names):
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings_dabias_output[i],
        X=man,
        Y=woman,
        permutations=10000
    )
    print(f"SC-WEAT Gender Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")
    effect_size, p_val = sc_weat_effect_size(
        A=occupation_embeddings_dabias_output[i],
        X=european_american,
        Y=african_american,
        permutations=10000
    )
    print(f"SC-WEAT Race Effect Size for {occupation}: {effect_size}")
    print(f"SC-WEAT p-value for {occupation}: {p_val}")

effect_size, p_val = sc_weat_effect_size(
    A=doctor,
    X=male_doctor,
    Y=female_doctor,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=engineer,
    X=male_engineer,
    Y=female_engineer,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=software_developer,
    X=man,
    Y=woman,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

effect_size, p_val = sc_weat_effect_size(
    A=software_developer,
    X=male_software_developer,
    Y=female_software_developer,
    permutations=10000
)

print("SC-WEAT Effect Size:", effect_size)
print("SC-WEAT p-value:", p_val)

"""## Approach Validation with OASIS"""

import pandas as pd
oasis_df = pd.read_csv("anonymized_directory")
# Find the 5 highest and lowest valence_mean images
highest_valence = oasis_df.nlargest(20, "Valence_mean")
lowest_valence = oasis_df.nsmallest(20, "Valence_mean")

print("\n20 Highest Valence Mean Images:")
print(highest_valence[["Theme", "Valence_mean"]])

print("\n20 Lowest Valence Mean Images:")
print(lowest_valence[["Theme", "Valence_mean"]])

oasis_themes = ["lake", "beach", "firework", "rainbow", "penguin", "war", "tumor", "animal carcass", "garbage dump", "fire"]
oasis_embeddings = [lake, beach, fireworks, rainbow, penguin, war, tumor, animal_carcass, garbage_dump, fire] # This line is added

for i, theme in enumerate(oasis_themes):
    effect_size, p_val = sc_weat_effect_size(
        A=oasis_embeddings[i], # Changed from theme to oasis_embeddings[i]
        X=pleasant,
        Y=unpleasant,
        permutations=10000
    )
    print(f"SC-WEAT Effect Size for {theme}:", effect_size)
    print(f"SC-WEAT p-value for {theme}:", p_val)

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset (already on disk from earlier step)
df = pd.read_csv('anonymized_directory')

# Convert percentage strings to numeric values
df['Pct_Women'] = df['% Women'].str.rstrip('%').astype(float)
df['Pct_White'] = df['% White'].str.rstrip('%').astype(float)
df['Pct_Men'] = 100 - df['Pct_Women']

# Split data for race and gender analyses
race_df = df[df['Associated Attribute'].isin(['Black', 'White'])].copy()
gender_df = df[df['Associated Attribute'].isin(['Male', 'Female'])].copy()

# Calculate Pearson correlations
race_r, race_p = pearsonr(race_df['Pct_White'], race_df['Race Effect Size'])
gender_r, gender_p = pearsonr(gender_df['Pct_Men'], gender_df['Gender Effect Size'])

# Color mappings
race_colors = {'White': 'tab:blue', 'Black': 'tab:orange'}
gender_colors = {'Male': 'tab:blue', 'Female': 'tab:pink'}

# Plot for race
fig, ax = plt.subplots(figsize=(8, 6))
for attr in ['White', 'Black']:
    subset = race_df[race_df['Associated Attribute'] == attr]
    ax.scatter(subset['Pct_White'], subset['Race Effect Size'],
               label=attr, color=race_colors[attr])

ax.set_xlabel('% White (BLS)')
ax.set_ylabel('Race Effect Size (Cohen\'s d)')
ax.set_title('Race Bias in T2V Outputs: % White vs. Effect Size for Occupations')
ax.text(0.05, 0.95, f'Pearson r = {race_r:.2f}', transform=ax.transAxes,
        verticalalignment='top')
ax.legend(title='Associated Attribute')
plt.tight_layout()
plt.show()

# Plot for gender
fig, ax = plt.subplots(figsize=(8, 6))
for attr in ['Male', 'Female']:
    subset = gender_df[gender_df['Associated Attribute'] == attr]
    ax.scatter(subset['Pct_Men'], subset['Gender Effect Size'],
               label=attr, color=gender_colors[attr])

ax.set_xlabel('% Men')
ax.set_ylabel('Gender Effect Size (Cohen\'s d)')
ax.set_title('Gender Bias in T2V Outputs: % Man vs. Effect Size for Occupations')
ax.text(0.05, 0.95, f'Pearson r = {gender_r:.2f}', transform=ax.transAxes,
        verticalalignment='top')
ax.legend(title='Associated Attribute')
plt.tight_layout()
plt.show()




"""## Visualziations"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap, re
from pathlib import Path

F_AWARD      = Path("anonymized_directory")
F_OCCUPATION = Path("anonymized_directory")
OUT_AWARD    = Path("anonymized_directory")
OUT_OCCUP    = Path("anonymized_directory")

def two_line(text: str, width: int = 22) -> str:
    """
    Wrap to *exactly* two lines (unless manual override below).
    """
    if not text:
        return ""
    wrapped = textwrap.wrap(text, width=width, break_long_words=False)
    if len(wrapped) == 1:                       # try to split in half
        mid = len(text.split()) // 2 or 1
        pieces = text.split()
        wrapped = [" ".join(pieces[:mid]), " ".join(pieces[mid:])]
    elif len(wrapped) > 2:                      # collapse to two lines
        wrapped = [" ".join(wrapped[:-1]), wrapped[-1]]
    return "\n".join(wrapped[:2])

def multi_line_map(raw: str) -> str:
    """
    Manual overrides for tricky occupation names.
    """
    custom = {
        "Postal service worker ": "Postal\nService\nWorker",         # 3 lines
        "Elementary School Teachers": "Elem.\nSchool\nTeachers",    # 3 lines
        "Housekeeper": "House\nkeeper",
        "Security guard": "Security\nguard",
        "Postsecondary teacher": "Postsecondary\nTeacher",
    }
    return custom.get(raw, raw)

# 1) AWARD HEAT-MAP
df_award = pd.read_csv(F_AWARD)

# columns holding effect sizes
GENDER_COLS = [
    "SC-VEAT Gender Effect Size",
    "SC-VEAT Gender - Debias Effect Size",
    "SC-VEAT Gender - Debias (Output) Effect Size",
]
RACE_COLS = [
    "SC-VEAT Race Effect Size",
    "SC-VEAT Race - Debias Effect Size",
    "SC-VEAT Race - Debias (Output) Effect Size",
]

# make sure numeric
for c in GENDER_COLS + RACE_COLS:
    df_award[c] = pd.to_numeric(df_award[c], errors="coerce")

# tidy labels (drop “Prize”, wrap, and – for Econ – use 3 lines)
def clean_award(name: str) -> str:
    name = re.sub(r"\bPrize\b", "", name).strip()
    if "Economic" in name:              # force three-line split
        return "Nobel\nEconomic\nScience"
    return two_line(name)

award_labels = [clean_award(n) for n in df_award["Award Name"].dropna()]
gender_mat   = df_award[GENDER_COLS].values
race_mat     = df_award[RACE_COLS].values

# fixed colour scale
V_MIN, V_MAX = (-1.0, 1.0)
COND = ["Control", "Debias 1", "Debias 2"]

fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"wspace": 0.28})
fig.patch.set_facecolor("white")

panels = [
    ("Award – Gender", gender_mat, axes[0]),
    ("Award – Race",   race_mat,   axes[1]),
]

for title, mat, ax in panels:
    im = ax.imshow(mat, cmap="coolwarm", vmin=V_MIN, vmax=V_MAX, aspect="auto")
    ax.set_xticks(range(len(COND)))
    ax.set_xticklabels(COND, rotation=25, ha="right",
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(award_labels)))
    ax.set_yticklabels(award_labels, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                    shrink=0.8, ticks=np.linspace(V_MIN, V_MAX, 5))
cbar.set_label("Cohen's d")

fig.suptitle("Gender and Race Effect Sizes (Cohen's d) for Award Videos",
             fontsize=19, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_AWARD, dpi=300, bbox_inches="tight", facecolor="white")

print(f"Award heat-map saved → {OUT_AWARD.resolve()}")

# 2) OCCUPATION HEAT-MAP
df_occ = pd.read_csv(F_OCCUPATION)

# numeric conversion
for col in df_occ.columns:
    if re.match(r"Control(\.\d+)?|Debias \d(\.\d+)?", col):
        df_occ[col] = pd.to_numeric(df_occ[col], errors="coerce")

def occ_matrix(lbl_col, triplet):
    sub = df_occ[df_occ[lbl_col].notna()]
    labels = [multi_line_map(t) for t in sub[lbl_col]]
    labels = [two_line(l) if "\n" not in l else l for l in labels]  # wrap others
    mat    = sub[triplet].values
    return mat, labels

mat_male,   lab_male   = occ_matrix("Male Associated",   ["Control","Debias 1","Debias 2"])
mat_female, lab_female = occ_matrix("Female Associated", ["Control.1","Debias 1.1","Debias 2.1"])
mat_white,  lab_white  = occ_matrix("White Associated",  ["Control.2","Debias 1.2","Debias 2.2"])
mat_black,  lab_black  = occ_matrix("Black Associated",  ["Control.3","Debias 1.3","Debias 2.3"])

fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                         gridspec_kw={"wspace": 0.3, "hspace": 0.35})
fig.patch.set_facecolor("white")

PANELS = [
    ("Male-associated occupations",   mat_male,   lab_male,   axes[0, 0]),
    ("Female-associated occupations", mat_female, lab_female, axes[0, 1]),
    ("White-associated occupations",    mat_white,  lab_white,  axes[1, 0]),
    ("Black-associated occupations",    mat_black,  lab_black,  axes[1, 1]),
]

for title, mat, labels, ax in PANELS:
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(COND)))
    ax.set_xticklabels(COND, rotation=25, ha="right",
                       fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold")
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                    shrink=0.8, ticks=np.linspace(-2.5, 2.5, 6))
cbar.set_label("Cohen's d", fontsize=12)

fig.suptitle("Gender and Race Effect Sizes (Cohen's d) for Occupation Videos",
             fontsize=19, fontweight="bold", y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_OCCUP, dpi=300, bbox_inches="tight", facecolor="white")


print(f"Occupation heat-map saved → {OUT_OCCUP.resolve()}")

