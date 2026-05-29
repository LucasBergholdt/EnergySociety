from dataclasses import dataclass, field, replace
import numpy as np
from aenum import Enum
from datasets import load_dataset


class Difficulty(Enum):
    _init_ = 'reward string' # For EASY: 200 can be accessed as .reward and 'easy' as .string
    
    EASY = 200, 'easy'
    MEDIUM = 500, 'medium'
    HARD = 800, 'hard'


@dataclass
class Job:
    """A job that agents can attempt"""
    job_id: str
    difficulty: Difficulty
    description: str
    answer: str
    category: str
    options: list[str] = field(default_factory=list)
    
    @property
    def reward(self) -> float:
        return self.difficulty.reward
    
    
    def prompt_format(self) -> str:
        """Formats the job to a string suitable for prompt"""
        formatted_options = ""
        if self.options: # [] evaluates to false
            formatted_options = "\n".join(f"  {i}. {o}" for i, o in enumerate(self.options))
        return f"Description:\n{self.description}\nOptions:\n{formatted_options}"
        
        
    def evaluate(self, agent_response: str) -> bool:
        """Evaluates if given agent_response is the correct answer to the job"""
        response = agent_response.lower()
        answer = self.answer.lower()
        
        if answer in response:
            return True
        
        # OBS: This returns true if the option number appears ANYWHERE in the agents response. Fragile.
        if self.options: # [] evaluates to false
            try:
                # Find which option the correct answer is:
                idx = self.options.index(self.answer) # may raise ValueError
                if str(idx) in response:
                    return True
            except ValueError:
                pass
        
        return False
    


class JobGenerator:
    """
    Stores a pool of jobs and picks samples from them for the round based on the difficulty distribution.
    """ 
    
    def __init__(self, jobs: list[Job], seed: int | None = None):
        # Reproducible randomness
        self.rng = np.random.default_rng(seed)
        # Create a jobs attribute that stores all jobs in groups by difficulty.
        self.jobs: dict[Difficulty, list[Job]] = {d: [] for d in Difficulty} # {"easy": [], "medium": [], "hard": []}
        # Fill difficulty groups with all the jobs of that difficulty:
        for job in jobs:
            self.jobs[job.difficulty].append(job)
            
    def generate_jobs(self, num_jobs: int, difficulty_distribution: dict[Difficulty, int]) -> list[Job]:
        """
        Requirement: num_jobs <= sum(difficulty_distribution.values())
        Picks num_jobs items from a pool of jobs specified by the difficulty distribution.
        
        For instance with:
        "difficulty_distribution": {
            "easy": 2, 
            "medium": 2,
            "hard": 2
        }
        
        num_jobs = 3
        
        3 jobs would be chosen from the 6 jobs available (2 of each difficulty).
        
        For chosing exact amount of jobs for each difficulty set num_jobs = sum of difficulty distribution values.
        E.g:
        "difficulty_distribution": {
            "easy": 3, 
            "medium": 2,
            "hard": 1
        }
        
        num_jobs = 6
        
        This will give 6 jobs in total: 3 easy jobs, 2 medium, 1 hard.
        """
        total_to_select = sum(difficulty_distribution.values())
        if num_jobs > total_to_select:
            raise ValueError(f"num_jobs ({num_jobs}) exceeds total jobs specified in difficulty_distribution ({total_to_select})")
        
        chosen_jobs = []
        job_counter = 0
        
        for difficulty, count in difficulty_distribution.items():
            # Select only the jobs of the difficulty we are currently sampling:
            job_pool = self.jobs[difficulty]
            # Ensure any jobs of this difficulty exists
            if not job_pool:
                raise ValueError(f"No jobs available for difficulty {difficulty}")
            
            # Enure we have enough unique jobs for this difficulty:
            if count > len(job_pool):
                raise ValueError(f"Requested {count} {difficulty} jobs, but only {len(job_pool)} unique jobs available.")
            
            idxs = self.rng.choice(len(job_pool), size=count, replace=False)
            selected = [job_pool[i] for i in idxs]
            for job in selected:
                job_unique_id = replace(job, job_id=f"job_{job_counter}")
                chosen_jobs.append(job_unique_id)
                job_counter += 1
        
        self.rng.shuffle(chosen_jobs)
        chosen_jobs = chosen_jobs[:num_jobs]
        return [replace(job, job_id=f"job_{i}") for i, job in enumerate(chosen_jobs)]
            





# ------------ loading datasets -----------------
        
def load_mmlu_pro_stratisfied_jobs() -> list[Job]:
    """
    Loads the MMLU-Pro-Stratified dataset and maps it to the easy, medium, hard
    difficulty levels in the Energy Society.
    """
    # Load dataset: https://huggingface.co/datasets/SunriserFuture/MMLU-Pro-Stratified/blob/main/README.md
    dataset = load_dataset("SunriserFuture/MMLU-Pro-Stratified")["train"]
    
    # Mapping environment difficulties to MMLU-Pro-Stratified difficulties
    difficulty_tags = {
        Difficulty.EASY: ["-----", "----", "---"],
        Difficulty.MEDIUM: ["--", "-", "+", "++"],
        Difficulty.HARD: ["+++++", "++++", "+++"],
    }
    
    # TODO: Could probably be more efficient with a single pass instead.
    jobs = []
    for difficulty, tags in difficulty_tags.items():
        for job in dataset.filter(lambda x: x["difficulty"] in tags):
            jobs.append(Job(
                job_id=job["question_id"],
                difficulty=difficulty,
                description=job["question"],
                answer=job["options"][job["answer_index"]],
                category=job["category"],
                options=job["options"],
            ))
        
    return jobs


# WAS USED TO TEST WHAT HAPPENS WITH EASIER JOBS, BUT IS CURRENTLY UNUSED
def load_easier_mmlu_pro_stratisfied_jobs() -> list[Job]:
    """Version where jobs are not as hard"""
    dataset = load_dataset("SunriserFuture/MMLU-Pro-Stratified")["train"]
    
    # Mapping environment difficulties to MMLU-Pro-Stratified difficulties
    difficulty_tags = {
        Difficulty.EASY: ["-----", "----"],
        Difficulty.MEDIUM: ["--", "-"],
        Difficulty.HARD: ["+++", "++"],
    }
    
    # TODO: Could probably be more efficient with a single pass instead.
    jobs = []
    for difficulty, tags in difficulty_tags.items():
        for job in dataset.filter(lambda x: x["difficulty"] in tags):
            jobs.append(Job(
                job_id=job["question_id"],
                difficulty=difficulty,
                description=job["question"],
                answer=job["options"][job["answer_index"]],
                category=job["category"],
                options=job["options"],
            ))
        
    return jobs