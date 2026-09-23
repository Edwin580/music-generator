# ruff: noqa
# fmt: off
"""Original Colab notebook: "Music Generation Neural Network Model".

Author: Edwin Cortazo - COGS-319, Fall 2024.
Kept verbatim for reference; see ../README.md for what was fixed in the `musegen` package.
"""

# %% [cell 1] Music Sequence Visualization and Analysis
import pandas as pd
import numpy as np
from plotnine import *

def create_piano_roll_plot(sequence, title="Piano Roll Visualization", max_steps=100):
    """Visualize the piano roll matrix of the sequence."""
    # Take only first max_steps for visibility
    sequence = sequence[:max_steps]

    # Create empty df to store the note data
    notes = []
    for time_step, note_vector in enumerate(sequence):
        active_notes = np.where(note_vector > 0)[0] # get indexes of active notes
        for note in active_notes:
            #store active notes
            notes.append({
                'time_step': time_step,
                'pitch': note
            })
    # convert list into pandas df for plotting
    df = pd.DataFrame(notes)

    # return none if no notes are active - NOTHING to plot
    if len(df) == 0:
        return None

    # Create the plot
    plot = (ggplot(df, aes(x='time_step', y='pitch'))
        + geom_point(color='#2563eb', size=3)
        + labs(title=title,
               x='Time Step',
               y='MIDI Pitch')
        + theme_minimal()
        + theme(figure_size=(15, 8))
        + scale_y_continuous(breaks=range(0, 128, 12))  # Show octave marks every 12 notes
    )

    return plot

def analyze_note_distribution(original_sequence, generated_sequence, max_notes=1000):
    """Compare note distributions between original and generated sequence"""
    # Get active notes
    def get_note_counts(sequence, max_notes=1000):
        sequence = sequence[:max_notes] # sequence is limited to max_notes
        note_counts = np.sum(sequence, axis=0) # sums activation for each note
        return pd.DataFrame({
            'pitch': range(128), # default pitch range for MIDI
            'count': note_counts # num of activatoins for each note / pitch
        })

    # note count for original sequence
    # -> NOTE: orignal sequence is referring to the FIRST sample song used, NOT all of them, it was just easier for visualization
    orig_df = get_note_counts(original_sequence)
    orig_df['source'] = 'Original'

    # note count for generated sequence
    gen_df = get_note_counts(generated_sequence)
    gen_df['source'] = 'Generated'

    # combines data from both into single df
    df = pd.concat([orig_df, gen_df])

    # creates plot comparing note distribution
    plot = (ggplot(df, aes(x='pitch', y='count', fill='source'))
        + geom_bar(stat='identity', position='dodge', alpha=0.7)
        + labs(title='Note Distribution Comparison',
               x='MIDI Pitch',
               y='Count',
               fill='Source')
        + theme_minimal()
        + theme(figure_size=(15, 6))
        + scale_x_continuous(breaks=range(0, 128, 12))  # show octave marks on x-axis
        + scale_fill_manual(values=['#2563eb', '#dc2626']) # BLUE for original, RED for generated
    )

    return plot

def plot_training_history(history):
    """create loss and accuracy plots for the training history"""
    # convert history dictionary into a df
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = np.arange(1, len(history_df) + 1) #add epoch nums

    # Reshape data for plotting
    df_metrics = pd.melt(history_df,
                        id_vars=['epoch'],
                        value_vars=['loss', 'val_loss', 'accuracy', 'val_accuracy'],
                        var_name='metric',
                        value_name='value')

    # seperate data into loss and accuracy df's
    df_loss = df_metrics[df_metrics['metric'].isin(['loss', 'val_loss'])]
    df_accuracy = df_metrics[df_metrics['metric'].isin(['accuracy', 'val_accuracy'])]

    # Create loss plot
    loss_plot = (ggplot(df_loss, aes(x='epoch', y='value', color='metric'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + labs(title='Model Loss During Training',
               x='Epoch',
               y='Loss',
               color='Metric')
        + theme_minimal()
        + scale_color_manual(values=['#2563eb', '#dc2626'],
                           labels=['Training Loss', 'Validation Loss'])
        + theme(figure_size=(12, 6),
               plot_title=element_text(size=14, face='bold'),
               axis_title=element_text(size=12))
    )

    # Create accuracy plot
    accuracy_plot = (ggplot(df_accuracy, aes(x='epoch', y='value', color='metric'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + labs(title='Model Accuracy During Training',
               x='Epoch',
               y='Accuracy',
               color='Metric')
        + theme_minimal()
        + scale_color_manual(values=['#16a34a', '#9333ea'],
                           labels=['Training Accuracy', 'Validation Accuracy'])
        + theme(figure_size=(12, 6),
               plot_title=element_text(size=14, face='bold'),
               axis_title=element_text(size=12))
    )

    return loss_plot, accuracy_plot


def create_all_visualizations(history, X_train, generated):
    """Create and save all visualizations"""
    # Training history plots
    loss_plot, accuracy_plot = plot_training_history(history)

    # Piano roll plots
    original_piano_roll = create_piano_roll_plot(X_train[0],
                                               title="Original Music Sample Piano Roll")
    # Generated roll plot
    generated_piano_roll = create_piano_roll_plot(generated,
                                                title="Generated Music Piano Roll")

    # Note distribution comparison
    note_dist = analyze_note_distribution(X_train[0], generated)

    # Save all plots as images; It was just easier to store and reference this way
    loss_plot.save('output/training_loss.png', dpi=300)
    accuracy_plot.save('output/training_accuracy.png', dpi=300) # width limits so they display properly
    if original_piano_roll:
        original_piano_roll.save('output/original_piano_roll.png', dpi=300)
    if generated_piano_roll:
        generated_piano_roll.save('output/generated_piano_roll.png', dpi=300)
    note_dist.save('output/note_distribution.png', dpi=300)

    return (loss_plot, accuracy_plot, original_piano_roll,
            generated_piano_roll, note_dist)


# %% [cell 2] Data processing, model, training and generation
import numpy as np
import tensorflow as tf
import pretty_midi
import glob
import os
from tensorflow.keras.layers import Input, SimpleRNN, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# sets up logging to track progress and issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MIDIConfig:
    """Configuration for MIDI processing - paramaters for how MIDI files are processed"""
    steps_per_beat: int = 4
    sequence_length: int = 48 # length of each sequence used for training
    velocity_threshold: int = 0 # trheshold to filter out noise
    min_note_duration: float = 0.125 # minimum duration for a note to be considered
    max_sequence_length: int = 2000 # increased from 1000 to handle longer sequences

class MIDIProcessor:
    def __init__(self, config: MIDIConfig = MIDIConfig()):
        """Initialize the MIDIProcessor with a configuration."""
        self.config = config

    def extract_monophonic_melody(self, midi_data: pretty_midi.PrettyMIDI) -> Optional[np.ndarray]:
        """Extract a monophonic melody from a MIDI file."""
        try:
            if not midi_data.instruments:
                return None # if no instruments are found in the MIDI file

            # choose instrument with highest average pitch to represent melody
            melody_instrument = max(midi_data.instruments,
                                  key=lambda x: np.mean([note.pitch for note in x.notes]))
            piano_roll = melody_instrument.get_piano_roll(fs=self.config.steps_per_beat)

            #converts piano roll into binary sequence
            note_sequence = (piano_roll > self.config.velocity_threshold).astype(np.float32)
            return note_sequence.T # transpose to make time steps as rows and notes as column

        except Exception as e:
            logger.error(f"Error extracting melody: {str(e)}")
            return None

    def create_training_sequences(self, note_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences from a note sequence."""
        x_sequences = []
        y_sequences = []
        sequence_length = min(len(note_sequence), self.config.max_sequence_length)

        # creates sequences of notes for training
        for i in range(0, sequence_length - self.config.sequence_length):
            x_seq = note_sequence[i:i + self.config.sequence_length]
            y_seq = note_sequence[i + 1:i + self.config.sequence_length + 1] #predict next note
            x_sequences.append(x_seq)
            y_sequences.append(y_seq)

        return np.array(x_sequences), np.array(y_sequences)

    def process_midi_dataset(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a dataset of MIDI files."""
        all_x_sequences = []
        all_y_sequences = []

        # lists all '.mid' files in the folder
        midi_files = glob.glob(os.path.join(folder_path, "*.mid"))
        logger.info(f"Found {len(midi_files)} MIDI files")

        # process each mid file
        for midi_file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                note_sequence = self.extract_monophonic_melody(midi_data)
                if note_sequence is None:
                    continue # skip if no valid melody was found

                x_seq, y_seq = self.create_training_sequences(note_sequence)
                if len(x_seq) > 0:  # Only add if sequences were generated
                  all_x_sequences.append(x_seq)
                  all_y_sequences.append(y_seq)
                  logger.info(f"Processed {midi_file}: Generated {len(x_seq)} sequences")

            except Exception as e:
                logger.error(f"Error processing {midi_file}: {str(e)}")
                continue
        # valid sequence check
        if not all_x_sequences:
            raise ValueError("No valid sequences were generated from the MIDI files")

        # combines sequences
        X = np.concatenate(all_x_sequences, axis=0)
        y = np.concatenate(all_y_sequences, axis=0)

        # shuffles data fro training
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        logger.info(f"Final dataset shape: {X.shape}")
        return X, y

class MusicRNN:
    def __init__(self, sequence_length=24, n_hidden=256):
        """Initialize the MusicRNN model."""
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.n_features = 128 # num of unique notes
        self.model = None

    def build_model(self):
        """Build the RNN model using LSTM layers."""
        inputs = Input(shape=(None, self.n_features))
        hidden = LSTM(self.n_hidden, return_sequences=True)(inputs)
        hidden = Dropout(0.3)(hidden) #dropout for regularization
        outputs = Dense(self.n_features, activation='sigmoid')(hidden)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return self.model

    def train(self, x_train, y_train, epochs=10, batch_size=128):
        """Train the RNN model."""
        if self.model is None:
            self.build_model()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3, # stop after 3 epochs of no improvement
            min_delta=0.0001, #minimal improvement threshold
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        return history

    def generate_sequence(self, seed_sequence, length=100, temperature=1.0):
        """Generate a new sequence based on a seed."""
        if len(seed_sequence.shape) == 2:
            seed_sequence = np.expand_dims(seed_sequence, 0) # ensures 3d for model

        generated_sequence = seed_sequence[0]

        # generates notes one at a time
        for _ in range(length):
            x = generated_sequence[-self.sequence_length:] # uses last part of the sequence
            x = np.expand_dims(x, 0) #expand current dimensions to match model shape

            # predict next note probabilites
            pred = self.model.predict(x, verbose=0)[0][-1]

            # applies temperature to control randomness (higher temperature = more random)
            pred = np.log(pred) / temperature
            pred = np.exp(pred) / np.sum(np.exp(pred))

            # sample from the probability distribution to select the next note
            next_notes = (np.random.random(pred.shape) < pred).astype(np.float32)
            generated_sequence = np.vstack([generated_sequence, next_notes])

        return generated_sequence

    def save_sequence_as_midi(self, sequence, output_file, tempo=120):
        """Save the generated sequence as a MIDI file"""
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # convert sequence into MIDI notes and add them to piano
        for time_step, note_vector in enumerate(sequence):
            active_notes = np.where(note_vector > 0)[0] # find active notes

            for note in active_notes:
                note_start = time_step * 0.25
                note_end = (time_step + 1) * 0.25
                note = pretty_midi.Note(
                    velocity=100, #loudness
                    pitch=int(note),
                    start=note_start, # node start time
                    end=note_end #node end time
                )
                piano.notes.append(note)

        pm.instruments.append(piano)
        pm.write(output_file)

def create_output_dirs():
    """Create output directories if they don't exist."""
    Path("output").mkdir(exist_ok=True)
    Path("output/midi").mkdir(exist_ok=True)
    Path("data/midi").mkdir(parents=True, exist_ok=True)

def download_sample_midi():
    """Download sample MIDI files if they don't exist."""
    try:
        import urllib.request

        midi_urls = [
            f"https://www.midiworld.com/download/{i}"
            for i in [3832, 3833, 3834, 4537, 4538, 4539, 4540, 4541,
                      3835, 3836, 3837, 4542, 4543, 4544, 4545, 4546,
                      3838, 3839, 3840, 4547, 4548, 4549, 4550,
                      4551, 4552, 4553, 4554, 4555, 4556,
                      4557, 4558, 4559, 4560, 4561,
                      4562, 4563, 4564, 4565,
                      # others
                      4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582,
                      4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592]
        ]  # (condensed from the notebook's explicit list of 58 URLs)

        for i, url in enumerate(midi_urls):
            output_file = f"data/midi/sample_{i}.mid"
            if not os.path.exists(output_file):
                logger.info(f"Downloading {url} to {output_file}")
                urllib.request.urlretrieve(url, output_file)

    except Exception as e:
        logger.error(f"Error downloading sample MIDI files: {str(e)}")
        logger.info("Please manually download some MIDI files to the data/midi directory")

def main():
    # create directories and download samples
    create_output_dirs()
    download_sample_midi()

    # initialize
    config = MIDIConfig(
        steps_per_beat=4,
        sequence_length=48,
        velocity_threshold=0
    )

    processor = MIDIProcessor(config)
    model = MusicRNN(
        sequence_length=48,
        n_hidden=256
    )

    # process data
    logger.info("Processing MIDI files...")
    try:
        X_train, y_train = processor.process_midi_dataset("data/midi")
    except ValueError as e:
        logger.error(str(e))
        return

    # Train model
    logger.info("Training model...")
    history = model.train(X_train, y_train, epochs=10, batch_size=128)

    # Generate new sequence
    logger.info("Generating new sequence...")
    seed_sequence = X_train[0:1]
    generated = model.generate_sequence(
        seed_sequence,
        length=100,
        temperature=1.0
    )

    # Create and save training visualizations
    logger.info("Creating visualizations...")
    plots = create_all_visualizations(history, X_train, generated)

    # Save generated sequence
    output_file = "output/midi/generated_sequence.mid"
    logger.info(f"Saving generated sequence to {output_file}")
    model.save_sequence_as_midi(generated, output_file)

    logger.info("Process complete!")
    logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Generated MIDI saved to: {output_file}")

if __name__ == "__main__":
    main()

# Original training log (Colab, 10 epochs, 273 batches/epoch, ~190 s/epoch):
#   ERROR: Error processing data/midi/sample_27.mid:
#   ERROR: Error processing data/midi/sample_28.mid: Could not decode key with 16 sharps and mode 1
#   Epoch  1/10 - accuracy: 0.0468 - loss: 0.1694 - val_accuracy: 0.0802 - val_loss: 0.0502
#   Epoch 10/10 - accuracy: 0.2499 - loss: 0.0255 - val_accuracy: 0.2541 - val_loss: 0.0234


# %% [cell 3] Display saved plots
# from IPython.display import Image, display
# def display_training_plots():
#     for title, f in [("Training Loss Plot", 'output/training_loss.png'),
#                      ("Training Accuracy Plot", 'output/training_accuracy.png'),
#                      ("Original Music Sample Piano Roll", 'output/original_piano_roll.png'),
#                      ("Generated Music Piano Roll", 'output/generated_piano_roll.png'),
#                      ("Note Distribution Comparison", 'output/note_distribution.png')]:
#         print(title + ":"); display(Image(filename=f, width=400)); print("\n")
# display_training_plots()


# %% [cell 4-6] Verify the generated MIDI file
# print(os.path.exists("output/midi/generated_sequence.mid"))   # True
# midi_data = pretty_midi.PrettyMIDI('output/midi/generated_sequence.mid')
# for i, instrument in enumerate(midi_data.instruments):
#     print(f"Instrument {i}: {len(instrument.notes)} notes")      # Instrument 0: 143 notes
#     for note in instrument.notes:
#         print(f"Note: {note.pitch}, Start: {note.start}, End: {note.end}")


# %% [cell 7] Waveform plot
# waveform = midi_data.synthesize()
# data = pd.DataFrame({'Samples': np.arange(10000), 'Amplitude': waveform[:10000]})
# (ggplot(data, aes(x='Samples', y='Amplitude')) + geom_line(color="blue")
#  + ggtitle("Waveform") + xlab("Samples") + ylab("Amplitude") + theme_minimal())


# %% [cell 8] Audio playback
# import soundfile as sf
# from IPython.display import Audio
# sf.write('output_test.mp3', waveform, 44100)   # NB: writes WAV samples to a .mp3 filename
# Audio('output_test.mp3')
