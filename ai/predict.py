"""
Hybrid GA-PSO Timetable Optimization
For: AI and Soft Computing Research Paper
Author: [Your Name]
University: Manipal University Jaipur
"""


import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from copy import deepcopy


# ============================================
# PART 0: AUTO-MERGE JOINT SLOTS
# ============================================

def auto_merge_joint_slots(df):
    """Merge joint class slots (same course, time, room, faculty but different sections)"""
    group_cols = ['Day', 'StartTime', 'EndTime', 'CourseCode', 'CourseName', 'course_type', 'Room', 'FacultyName']
    grouped = df.groupby(group_cols)
    merged_rows = []
    for idx, group in grouped:
        base_row = group.iloc[0].copy()
        combined_sections = ','.join(sorted(group['Section'].unique()))
        base_row['CombinedSections'] = combined_sections
        base_row['Combined'] = len(group['Section'].unique()) > 1
        merged_rows.append(base_row)
    merged_df = pd.DataFrame(merged_rows)
    return merged_df


# ============================================
# PART 1: LOAD DATA
# ============================================


print("Loading data...")
courses_df = pd.read_csv('course.csv')
dse_timetable_df = pd.read_csv('dse_timetable.csv')
classrooms_df = pd.read_csv('room.csv')
faculty_df = pd.read_csv('faculty.csv')

# Apply auto-merge to get conflict-free timetable slots
timeslots_df = auto_merge_joint_slots(dse_timetable_df)

num_courses = len(courses_df)
num_timeslots = len(timeslots_df)
num_classrooms = len(classrooms_df)


print(f"✓ Loaded {num_courses} courses, {num_timeslots} timeslots (merged), {num_classrooms} classrooms")
print(f"✓ Joint sessions merged: {timeslots_df['Combined'].sum()}")


# ============================================
# PART 2: CHROMOSOME REPRESENTATION
# ============================================
# Chromosome = list of genes
# Each gene = [course_id, timeslot_id, classroom_id]
# Example: [[0, 18, 2], [1, 19, 5], [2, 20, 1], ...]


def create_random_timetable():
    """Create a random timetable (chromosome)"""
    timetable = []
    for course_id in range(num_courses):
        slot_id = random.randint(0, num_timeslots - 1)
        room_id = random.randint(0, num_classrooms - 1)
        timetable.append([course_id, slot_id, room_id])
    return timetable


# ============================================
# PART 3: CONSTRAINT CHECKING
# ============================================


def check_room_conflicts(timetable):
    """Count instances where same room is used at same time"""
    conflicts = 0
    slot_room_map = {}
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        key = (slot_id, room_id)
        
        if key in slot_room_map:
            conflicts += 1  # Room conflict!
        else:
            slot_room_map[key] = course_id
    
    return conflicts


def check_faculty_conflicts(timetable):
    """Count instances where same faculty teaches multiple courses at same time"""
    conflicts = 0
    slot_faculty_map = {}
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        faculty = courses_df.iloc[course_id]['Faculty']
        key = (slot_id, faculty)
        
        if key in slot_faculty_map:
            conflicts += 1  # Faculty conflict!
        else:
            slot_faculty_map[key] = course_id
    
    return conflicts


def check_student_conflicts(timetable):
    """Count instances where same section has multiple courses at same time"""
    conflicts = 0
    slot_section_map = {}
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        section = courses_df.iloc[course_id]['Section']
        key = (slot_id, section)
        
        if key in slot_section_map:
            conflicts += 1  # Student conflict!
        else:
            slot_section_map[key] = course_id
    
    return conflicts


def check_lab_room_constraints(timetable):
    """Check if lab courses are assigned to lab rooms"""
    violations = 0
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        course_needs_lab = courses_df.iloc[course_id]['HasLab'] == 'Yes'
        room_has_lab = classrooms_df.iloc[room_id]['HasLab'] == 'Yes'
        
        if course_needs_lab and not room_has_lab:
            violations += 1  # Lab course in non-lab room!
    
    return violations


def count_student_gaps(timetable):
    """Count gaps in student schedules (soft constraint)"""
    gaps = 0
    section_schedules = {}
    
    # Build schedule for each section
    for gene in timetable:
        course_id, slot_id, room_id = gene
        section = courses_df.iloc[course_id]['Section']
        day = timeslots_df.iloc[slot_id]['Day']
        
        key = (section, day)
        if key not in section_schedules:
            section_schedules[key] = []
        section_schedules[key].append(slot_id)
    
    # Count gaps for each section per day
    for key, slots in section_schedules.items():
        if len(slots) > 1:
            slots_sorted = sorted(slots)
            for i in range(len(slots_sorted) - 1):
                gap = slots_sorted[i+1] - slots_sorted[i] - 1
                if gap > 0:
                    gaps += gap
    
    return gaps


def count_unpopular_slots(timetable):
    """Penalize use of unpopular time slots"""
    penalty = 0
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        is_popular = timeslots_df.iloc[slot_id].get('IsPopular', 'Yes')
        
        if is_popular == 'No':
            penalty += 1
    
    return penalty


# ============================================
# PART 4: FITNESS FUNCTION
# ============================================


def evaluate_fitness(timetable):
    """
    Calculate fitness score (lower penalty = better)
    Hard constraints have high penalties
    Soft constraints have low penalties
    """
    penalty = 0
    
    # Hard constraints (MUST be satisfied)
    penalty += check_room_conflicts(timetable) * 100
    penalty += check_faculty_conflicts(timetable) * 100
    penalty += check_student_conflicts(timetable) * 100
    penalty += check_lab_room_constraints(timetable) * 50
    
    # Soft constraints (should be minimized)
    penalty += count_student_gaps(timetable) * 5
    penalty += count_unpopular_slots(timetable) * 2
    
    return -penalty  # Negative because higher (closer to 0) is better


# ============================================
# PART 5: GENETIC ALGORITHM OPERATIONS
# ============================================


def tournament_selection(population, fitness_scores, k=3):
    """Select best individual from k random ones"""
    selected = random.sample(list(zip(population, fitness_scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)  # Higher fitness is better
    return deepcopy(selected[0][0])


def crossover(parent1, parent2):
    """Single-point crossover"""
    if random.random() < 0.8:  # 80% crossover rate
        point = random.randint(1, num_courses - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return deepcopy(parent1), deepcopy(parent2)


def mutate(timetable, mutation_rate=0.1):
    """Randomly change some assignments"""
    mutated = deepcopy(timetable)
    
    for i in range(num_courses):
        if random.random() < mutation_rate:
            # Mutate this gene
            if random.random() < 0.5:
                # Change timeslot
                mutated[i][1] = random.randint(0, num_timeslots - 1)
            else:
                # Change classroom
                mutated[i][2] = random.randint(0, num_classrooms - 1)
    
    return mutated


# ============================================
# PART 6: GENETIC ALGORITHM
# ============================================


def run_genetic_algorithm(pop_size=50, generations=100, verbose=True):
    """Run GA and return best solution"""
    
    if verbose:
        print("\n" + "="*50)
        print("RUNNING GENETIC ALGORITHM")
        print("="*50)
    
    # Initialize population
    population = [create_random_timetable() for _ in range(pop_size)]
    best_fitness_history = []
    
    start_time = time.time()
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_fitness_history.append(best_fitness)
        
        if verbose and gen % 10 == 0:
            print(f"Generation {gen}: Best Fitness = {best_fitness}")
        
        # Create next generation
        new_population = []
        
        # Elitism: Keep best 2 individuals
        sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        new_population.extend([deepcopy(sorted_pop[0][0]), deepcopy(sorted_pop[1][0])])
        
        # Generate rest through selection, crossover, mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
    
    # Return best solution
    fitness_scores = [evaluate_fitness(ind) for ind in population]
    best_idx = np.argmax(fitness_scores)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\nGA Complete!")
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Best Fitness: {fitness_scores[best_idx]}")
    
    return population[best_idx], best_fitness_history, elapsed_time


# ============================================
# PART 7: PARTICLE SWARM OPTIMIZATION
# ============================================


def pso_optimize(initial_solution, iterations=50, num_particles=20, verbose=True):
    """Refine solution using PSO"""
    
    if verbose:
        print("\n" + "="*50)
        print("RUNNING PARTICLE SWARM OPTIMIZATION")
        print("="*50)
    
    start_time = time.time()
    
    # Initialize particles around initial solution
    particles = []
    velocities = []
    personal_best = []
    personal_best_fitness = []
    
    for _ in range(num_particles):
        particle = deepcopy(initial_solution)
        # Add small random perturbation
        particle = mutate(particle, mutation_rate=0.2)
        particles.append(particle)
        velocities.append([[0, 0, 0] for _ in range(num_courses)])
        personal_best.append(deepcopy(particle))
        personal_best_fitness.append(evaluate_fitness(particle))
    
    # Global best
    global_best = deepcopy(initial_solution)
    global_best_fitness = evaluate_fitness(initial_solution)
    
    fitness_history = [global_best_fitness]
    
    # PSO parameters
    w = 0.7  # Inertia
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    
    for iteration in range(iterations):
        for i in range(num_particles):
            # Update velocity and position
            for j in range(num_courses):
                # Velocity update (simplified for discrete space)
                if random.random() < c1 * random.random():
                    # Move toward personal best
                    particles[i][j] = deepcopy(personal_best[i][j])
                
                if random.random() < c2 * random.random():
                    # Move toward global best
                    particles[i][j] = deepcopy(global_best[j])
                
                # Random exploration
                if random.random() < 0.1:
                    particles[i][j][1] = random.randint(0, num_timeslots - 1)
                    particles[i][j][2] = random.randint(0, num_classrooms - 1)
            
            # Evaluate fitness
            fitness = evaluate_fitness(particles[i])
            
            # Update personal best
            if fitness > personal_best_fitness[i]:
                personal_best[i] = deepcopy(particles[i])
                personal_best_fitness[i] = fitness
            
            # Update global best
            if fitness > global_best_fitness:
                global_best = deepcopy(particles[i])
                global_best_fitness = fitness
        
        fitness_history.append(global_best_fitness)
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Fitness = {global_best_fitness}")
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\nPSO Complete!")
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Best Fitness: {global_best_fitness}")
    
    return global_best, fitness_history, elapsed_time


# ============================================
# PART 8: HYBRID GA-PSO
# ============================================


def run_hybrid_ga_pso(ga_generations=100, pso_iterations=50):
    """Run hybrid approach: GA followed by PSO refinement"""
    
    print("\n" + "="*60)
    print("HYBRID GA-PSO TIMETABLE OPTIMIZATION")
    print("="*60)
    
    # Phase 1: GA
    ga_solution, ga_history, ga_time = run_genetic_algorithm(generations=ga_generations)
    
    # Phase 2: PSO refinement
    final_solution, pso_history, pso_time = pso_optimize(ga_solution, iterations=pso_iterations)
    
    total_time = ga_time + pso_time
    
    print("\n" + "="*60)
    print("HYBRID APPROACH COMPLETE")
    print("="*60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Final Fitness: {evaluate_fitness(final_solution)}")
    
    return final_solution, ga_history, pso_history, total_time


# ============================================
# PART 9: DISPLAY RESULTS
# ============================================


def print_timetable(timetable, title="TIMETABLE"):
    """Print timetable in readable format"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)
    
    for gene in timetable:
        course_id, slot_id, room_id = gene
        
        course_name = courses_df.iloc[course_id]['CourseName']
        course_code = courses_df.iloc[course_id]['CourseCode']
        day = timeslots_df.iloc[slot_id]['Day']
        start_time = timeslots_df.iloc[slot_id]['StartTime']
        end_time = timeslots_df.iloc[slot_id]['EndTime']
        room_name = classrooms_df.iloc[room_id]['RoomName']
        
        print(f"{course_code:10} | {course_name:40} | {day:10} | {start_time}-{end_time} | {room_name}")
    
    print("="*80)


def analyze_solution(timetable, label="Solution"):
    """Analyze and print constraint violations"""
    print(f"\n{'='*50}")
    print(f"{label} ANALYSIS")
    print(f"{'='*50}")
    print(f"Room Conflicts: {check_room_conflicts(timetable)}")
    print(f"Faculty Conflicts: {check_faculty_conflicts(timetable)}")
    print(f"Student Conflicts: {check_student_conflicts(timetable)}")
    print(f"Lab Room Violations: {check_lab_room_constraints(timetable)}")
    print(f"Student Gaps: {count_student_gaps(timetable)}")
    print(f"Unpopular Slots Used: {count_unpopular_slots(timetable)}")
    print(f"Overall Fitness: {evaluate_fitness(timetable)}")
    print(f"{'='*50}")


# ============================================
# PART 10: MAIN EXECUTION
# ============================================


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TIMETABLE OPTIMIZATION - RESEARCH PROJECT")
    print("Manipal University Jaipur - Data Science & Engineering")
    print("="*80)
    
    # Run experiments
    print("\n\nEXPERIMENT 1: GA Only")
    ga_solution, ga_history, ga_time = run_genetic_algorithm(generations=100)
    analyze_solution(ga_solution, "GA Only")
    
    print("\n\nEXPERIMENT 2: PSO Only")
    random_start = create_random_timetable()
    pso_solution, pso_history, pso_time = pso_optimize(random_start, iterations=100)
    analyze_solution(pso_solution, "PSO Only")
    
    print("\n\nEXPERIMENT 3: Hybrid GA-PSO (YOUR APPROACH)")
    hybrid_solution, ga_hist, pso_hist, hybrid_time = run_hybrid_ga_pso(ga_generations=100, pso_iterations=50)
    analyze_solution(hybrid_solution, "Hybrid GA-PSO")
    
    # Print best timetable
    print_timetable(hybrid_solution, "OPTIMIZED TIMETABLE (GA-PSO)")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ga_history, label='GA Only')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('GA Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    combined_history = ga_hist + pso_hist
    plt.plot(combined_history, label='Hybrid GA-PSO', color='green')
    plt.axvline(x=len(ga_hist), color='red', linestyle='--', label='PSO Starts')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    plt.title('Hybrid GA-PSO Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=300)
    print("\n✓ Convergence plots saved as 'convergence_plots.png'")
    
    # Create comparison table
    results_df = pd.DataFrame({
        'Algorithm': ['GA Only', 'PSO Only', 'Hybrid GA-PSO'],
        'Fitness Score': [
            evaluate_fitness(ga_solution),
            evaluate_fitness(pso_solution),
            evaluate_fitness(hybrid_solution)
        ],
        'Time (sec)': [ga_time, pso_time, hybrid_time],
        'Hard Violations': [
            check_room_conflicts(ga_solution) + check_faculty_conflicts(ga_solution) + check_student_conflicts(ga_solution),
            check_room_conflicts(pso_solution) + check_faculty_conflicts(pso_solution) + check_student_conflicts(pso_solution),
            check_room_conflicts(hybrid_solution) + check_faculty_conflicts(hybrid_solution) + check_student_conflicts(hybrid_solution)
        ]
    })
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv('comparison_results.csv', index=False)
    print("\n✓ Results saved as 'comparison_results.csv'")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - convergence_plots.png (for your paper)")
    print("  - comparison_results.csv (for your results table)")

# ============================================
# PART 11: PRINT SORTED TIMETABLES BY SECTION (WITH TEACHER NAME PER SECTION)
# ============================================

def print_section_timetables():
    # Use your original (not merged) timetable for accurate teachers/sections
    timetable_df = pd.read_csv('dse_timetable.csv')

    sections = timetable_df['Section'].unique()
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    for section in sections:
        print(f"\n{'='*70}")
        print(f"TIMETABLE FOR SECTION {section}")
        print(f"{'='*70}")
        df = timetable_df[timetable_df['Section'] == section].copy()
        # Convert StartTime to minutes for sorting
        df['StartMinutes'] = df['StartTime'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]))
        df['DayNum'] = df['Day'].map({d:i for i,d in enumerate(day_order)})
        df_sorted = df.sort_values(['DayNum', 'StartMinutes'])
        for _, row in df_sorted.iterrows():
            print(f"{row['Day']:10} {row['StartTime']:5}-{row['EndTime']:5} | {row['CourseCode']:10} | {row['CourseName']:40} | {row['Room']:10} | {row['FacultyName']}")
        print(f"{'='*70}")

# CALL THIS TO PRINT SECTION-WISE, TIME-ORDERED TIMETABLE AFTER ALL OPTIMIZATION
print_section_timetables()

