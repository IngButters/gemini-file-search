"""Interactive chat interface for the Gemini File Search application."""

from pathlib import Path
from datetime import datetime

from src.config import Config
from src.gemini_client import GeminiChatClient
from src.file_search_manager import FileSearchManager
from src.table_extractor import TableExtractor
from src.semantic_matcher import SemanticMatcher
from src.comparison_generator import ComparisonGenerator


class ChatInterface:
    """Interactive chat interface with file search store management."""

    def __init__(self):
        """Initialize the chat interface."""
        # Validate configuration
        Config.validate()

        # Initialize Gemini client
        self.gemini_client = GeminiChatClient(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.MODEL_NAME,
            system_instruction=Config.SYSTEM_INSTRUCTION,
            enable_thinking=Config.ENABLE_THINKING,
            thinking_budget=Config.THINKING_BUDGET
        )

        # Initialize file search manager
        self.file_search_manager = FileSearchManager(
            client=self.gemini_client.client,
            store_prefix=Config.FILE_SEARCH_STORE_PREFIX
        )

        # Initialize table comparison modules
        self.table_extractor = TableExtractor(gemini_client=self.gemini_client)
        self.semantic_matcher = SemanticMatcher(gemini_client=self.gemini_client)
        self.comparison_generator = ComparisonGenerator()

        self.current_store = None
        self.is_running = False

        # Table comparison state
        self.extracted_tables = {}  # filename -> DataFrame
        self.base_file = None
        self.comparison_files = []

    def start(self):
        """Start the chat interface."""
        self.is_running = True
        self.display_welcome()
        self.main_menu()

    def display_welcome(self):
        """Display welcome message."""
        print("\n" + "="*70)
        print("  GEMINI FILE SEARCH CHAT APPLICATION")
        print("="*70)
        print(f"\nModel: {Config.MODEL_NAME}")
        print(f"Files Directory: {Config.FILES_DIR}")
        print("\nType '/help' for available commands")
        print("Type '/quit' to exit")
        print("="*70)

    def main_menu(self):
        """Main interaction loop."""
        while self.is_running:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                else:
                    # Regular chat message
                    self.handle_chat_message(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self.is_running = False
            except Exception as e:
                print(f"\nError: {e}")

        print("\nGoodbye!")

    def handle_command(self, command: str):
        """Handle user commands.

        Args:
            command: Command string starting with '/'
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == '/help':
            self.show_help()
        elif cmd == '/quit' or cmd == '/exit':
            self.is_running = False
        elif cmd == '/create' or cmd == '/create-store':
            self.cmd_create_store(args)
        elif cmd == '/list' or cmd == '/list-stores':
            self.cmd_list_stores()
        elif cmd == '/select' or cmd == '/select-store':
            self.cmd_select_store(args)
        elif cmd == '/delete' or cmd == '/delete-store':
            self.cmd_delete_store(args)
        elif cmd == '/upload' or cmd == '/upload-files':
            self.cmd_upload_files()
        elif cmd == '/store' or cmd == '/store-info':
            self.cmd_store_info()
        elif cmd == '/start' or cmd == '/start-chat':
            self.cmd_start_chat()
        elif cmd == '/reset' or cmd == '/reset-chat':
            self.cmd_reset_chat()
        elif cmd == '/history':
            self.cmd_show_history()
        elif cmd == '/export' or cmd == '/export-chat':
            self.cmd_export_chat(args)
        # Table comparison commands
        elif cmd == '/extract' or cmd == '/extract-tables':
            self.cmd_extract_tables(args)
        elif cmd == '/preview' or cmd == '/preview-table':
            self.cmd_preview_table(args)
        elif cmd == '/set-base':
            self.cmd_set_base(args)
        elif cmd == '/compare' or cmd == '/compare-tables':
            self.cmd_compare_tables(args)
        elif cmd == '/export-comparison':
            self.cmd_export_comparison(args)
        else:
            print(f"Unknown command: {cmd}")
            print("Type '/help' for available commands")

    def handle_chat_message(self, message: str):
        """Handle regular chat messages.

        Args:
            message: User message
        """
        if not self.gemini_client.chat:
            print("\nPlease start a chat session first using '/start'")
            return

        if not self.current_store:
            print(
                "\nWarning: No file search store selected. Using chat without file search.")
            print("Use '/select <store-name>' to enable file search.")

        # Send message and display response
        response = self.gemini_client.send_message(message)
        if response:
            self.gemini_client.display_response(response)

    def show_help(self):
        """Display help information."""
        print("\n" + "="*70)
        print("AVAILABLE COMMANDS")
        print("="*70)
        print("\nFile Search Store Management:")
        print("  /create [name]           - Create a new file search store")
        print("  /list                    - List all file search stores")
        print("  /select <name>           - Select a store for chat queries")
        print("  /delete <name>           - Delete a file search store")
        print("  /upload                  - Upload files from 'files' directory")
        print("  /store                   - Show current store information")
        print("\nChat Commands:")
        print("  /start                   - Start a new chat session")
        print("  /reset                   - Reset the current chat session")
        print("  /history                 - Show chat history")
        print("  /export [filename]       - Export chat history as markdown")
        print("\nTable Comparison Commands:")
        print("  /extract [file]          - Extract tables from files in 'files' directory")
        print("  /preview <file>          - Preview extracted table")
        print("  /set-base <file>         - Set base file for comparison")
        print("  /compare <files...>      - Compare base with other files")
        print("  /export-comparison <out> - Export comparison to Excel/CSV")
        print("\nGeneral:")
        print("  /help                    - Show this help message")
        print("  /quit or /exit           - Exit the application")
        print(
            "\nNote: Commands support both short (/create) and long (/create-store) forms.")
        print("      To chat, simply type your message without a command prefix.")
        print("="*70)

    def cmd_create_store(self, display_name: str):
        """Create a new file search store.

        Args:
            display_name: Optional display name for the store
        """
        print("\nCreating file search store...")
        store = self.file_search_manager.create_store(
            display_name=display_name if display_name else None
        )

        if store:
            print(f"\nStore created successfully!")
            print(f"Store Name: {store.name}")

            # Ask if user wants to select this store
            choice = input(
                "\nSelect this store for chat? (y/n): ").strip().lower()
            if choice == 'y':
                self.current_store = store
                self.gemini_client.set_file_search_stores([store.name])
                print(f"Selected store: {store.name}")

    def cmd_list_stores(self):
        """List all file search stores."""
        self.file_search_manager.display_stores_summary()

    def cmd_select_store(self, store_name: str):
        """Select a file search store for chat.

        Args:
            store_name: Name of the store to select
        """
        if not store_name:
            print("\nError: Please provide a store name")
            print("Usage: /select <store-name>")
            return

        store = self.file_search_manager.get_store(store_name)
        if store:
            self.current_store = store
            self.gemini_client.set_file_search_stores([store.name])
            print(f"\nSelected store: {store.name}")
        else:
            print(f"\nStore not found: {store_name}")
            print("Use '/list' to see available stores")

    def cmd_delete_store(self, store_name: str):
        """Delete a file search store.

        Args:
            store_name: Name of the store to delete
        """
        if not store_name:
            print("\nError: Please provide a store name")
            print("Usage: /delete <store-name>")
            return

        # Confirm deletion
        print(f"\nAre you sure you want to delete '{store_name}'?")
        confirm = input("Type 'yes' to confirm: ").strip().lower()

        if confirm == 'yes':
            if self.file_search_manager.delete_store(store_name):
                # Deselect if this was the current store
                if self.current_store and self.current_store.name == store_name:
                    self.current_store = None
                    self.gemini_client.set_file_search_stores([])
                    print("Current store deselected.")
        else:
            print("Deletion cancelled.")

    def cmd_upload_files(self):
        """Upload files from the files directory to the current store."""
        if not self.current_store:
            print("\nError: No store selected. Please select a store first.")
            print("Use '/select <store-name>' or '/create'")
            return

        print(f"\nUploading files from: {Config.FILES_DIR}")
        print(f"To store: {self.current_store.name}")

        count = self.file_search_manager.upload_files_from_directory(
            Config.FILES_DIR,
            self.current_store.name
        )

        if count > 0:
            print(f"\nSuccessfully uploaded {count} file(s)!")
        else:
            print("\nNo files uploaded. Make sure files exist in the 'files' directory.")

    def cmd_store_info(self):
        """Show information about the current store."""
        if not self.current_store:
            print("\nNo store currently selected.")
            print("Use '/select <store-name>' to select a store")
            return

        print("\n" + "="*70)
        print("CURRENT STORE INFORMATION")
        print("="*70)
        print(f"\nStore Name: {self.current_store.name}")
        if hasattr(self.current_store, 'display_name'):
            print(f"Display Name: {self.current_store.display_name}")
        if hasattr(self.current_store, 'create_time'):
            print(f"Created: {self.current_store.create_time}")
        print("="*70)

    def cmd_start_chat(self):
        """Start a new chat session."""
        if self.gemini_client.chat:
            print("\nA chat session is already active.")
            choice = input(
                "Reset and start new session? (y/n): ").strip().lower()
            if choice != 'y':
                return

        if self.gemini_client.start_chat():
            if self.current_store:
                print(f"Using file search store: {self.current_store.name}")
            else:
                print(
                    "No file search store selected. Chat will work without file search.")
                print("Use '/select <store-name>' to enable file search.")

    def cmd_reset_chat(self):
        """Reset the chat session."""
        if not self.gemini_client.chat:
            print("\nNo active chat session to reset.")
            return

        self.gemini_client.reset_chat()

    def cmd_show_history(self):
        """Show chat history."""
        history = self.gemini_client.get_chat_history()

        if not history:
            print("\nNo chat history available.")
            return

        print("\n" + "="*70)
        print("CHAT HISTORY")
        print("="*70)

        for message in history:
            role = message.role.upper()
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"\n{role}: {part.text}")

        print("="*70)

    def cmd_export_chat(self, filename: str):
        """Export chat history as markdown file.

        Args:
            filename: Optional custom filename (without .md extension)
        """
        history = self.gemini_client.get_chat_history()

        if not history:
            print("\nNo chat history available to export.")
            return

        # Generate filename with timestamp if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}"

        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'

        # Create exports directory if it doesn't exist
        exports_dir = Path('exports')
        exports_dir.mkdir(exist_ok=True)

        filepath = exports_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header
                f.write("# Gemini Chat Conversation Export\n\n")
                f.write(
                    f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Model:** {Config.MODEL_NAME}\n\n")

                if self.current_store:
                    f.write(
                        f"**File Search Store:** {self.current_store.name}\n\n")

                f.write("---\n\n")

                # Write conversation
                for message in history:
                    role = message.role.upper()

                    if hasattr(message, 'parts') and message.parts:
                        for part in message.parts:
                            if hasattr(part, 'text') and part.text:
                                # Format based on role
                                if role == 'USER':
                                    f.write(f"## You\n\n{part.text}\n\n")
                                elif role == 'MODEL':
                                    f.write(f"## Assistant\n\n{part.text}\n\n")

                                # Add grounding metadata if available
                                if role == 'MODEL' and hasattr(message, 'candidates'):
                                    for candidate in message.candidates:
                                        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                            f.write(self._format_citations_markdown(
                                                candidate.grounding_metadata))

                                f.write("---\n\n")

            print(f"\nChat exported successfully to: {filepath}")

        except Exception as e:
            print(f"\nError exporting chat: {e}")

    def _format_citations_markdown(self, grounding_metadata) -> str:
        """Format grounding metadata as markdown.

        Args:
            grounding_metadata: Grounding metadata from response

        Returns:
            Formatted markdown string
        """
        markdown = "### Citations\n\n"

        # Display search queries if available
        if hasattr(grounding_metadata, 'search_entry_point') and grounding_metadata.search_entry_point:
            if hasattr(grounding_metadata.search_entry_point, 'rendered_content'):
                markdown += f"**Search queries used:** {grounding_metadata.search_entry_point.rendered_content}\n\n"

        # Display grounding chunks (sources)
        if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
            markdown += f"**Sources ({len(grounding_metadata.grounding_chunks)}):**\n\n"

            for i, chunk in enumerate(grounding_metadata.grounding_chunks, 1):
                markdown += f"{i}. "

                # Try to extract relevant information from the chunk
                if hasattr(chunk, 'web') and chunk.web:
                    title = chunk.web.title if hasattr(
                        chunk.web, 'title') else 'N/A'
                    markdown += f"**Web:** {title}\n"
                    if hasattr(chunk.web, 'uri'):
                        markdown += f"   - URI: {chunk.web.uri}\n"
                elif hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                    # For file search results
                    if hasattr(chunk.retrieved_context, 'uri'):
                        markdown += f"**Document:** {chunk.retrieved_context.uri}\n"
                    if hasattr(chunk.retrieved_context, 'title'):
                        markdown += f"   - **Title:** {chunk.retrieved_context.title}\n"

                markdown += "\n"

        # Display grounding supports
        if hasattr(grounding_metadata, 'grounding_supports') and grounding_metadata.grounding_supports:
            markdown += f"**Grounding supports:** {len(grounding_metadata.grounding_supports)} segment(s) grounded\n\n"

        return markdown

    # ========================================================================
    # Table Comparison Commands
    # ========================================================================

    def cmd_extract_tables(self, args: str):
        """Extract tables from files in 'files' directory.

        Args:
            args: Optional filename to extract from (or empty for all)
        """
        print("\n" + "="*60)
        print("EXTRACTING TABLES")
        print("="*60)

        # Get files to extract
        if args.strip():
            # Extract specific file
            filepath = Config.FILES_DIR / args.strip()
            if not filepath.exists():
                print(f"\nError: File not found: {filepath}")
                return
            files_to_extract = [filepath]
        else:
            # Extract all supported files
            files_to_extract = []
            for ext in ['.pdf', '.xlsx', '.xls', '.csv']:
                files_to_extract.extend(Config.FILES_DIR.glob(f'*{ext}'))

        if not files_to_extract:
            print("\nNo supported files found in 'files' directory.")
            print("Supported formats: PDF, Excel (.xlsx, .xls), CSV")
            return

        print(f"\nFound {len(files_to_extract)} file(s) to extract\n")

        # Extract tables from each file
        for filepath in files_to_extract:
            print(f"Extracting from: {filepath.name}...")

            df = self.table_extractor.extract_table(filepath)

            if df is not None:
                # Detect columns
                column_map = self.table_extractor.detect_columns(df)
                print(f"  Detected columns: {column_map}")

                # Normalize table
                normalized_df = self.table_extractor.normalize_table(df, column_map)

                # Store extracted table
                self.extracted_tables[filepath.name] = normalized_df

                # Show info
                info = self.table_extractor.get_table_info(normalized_df)
                print(f"  ✓ Extracted {info['num_items_with_prices']} items with prices")
            else:
                print(f"  ✗ Failed to extract table")

        print(f"\n✓ Extraction complete. {len(self.extracted_tables)} tables extracted.")
        print("\nUse '/preview <filename>' to see table contents")
        print("Use '/set-base <filename>' to set base file for comparison")

    def cmd_preview_table(self, args: str):
        """Preview extracted table.

        Args:
            args: Filename to preview
        """
        if not args.strip():
            print("\nError: Please specify a filename")
            print("Usage: /preview <filename>")
            print(f"\nAvailable files: {list(self.extracted_tables.keys())}")
            return

        filename = args.strip()

        if filename not in self.extracted_tables:
            print(f"\nError: Table not extracted for: {filename}")
            print(f"Available files: {list(self.extracted_tables.keys())}")
            print("\nUse '/extract' to extract tables first")
            return

        df = self.extracted_tables[filename]
        preview = self.table_extractor.preview_table(df, num_rows=10)

        print(f"\n{filename}")
        print(preview)

    def cmd_set_base(self, args: str):
        """Set base file for comparison.

        Args:
            args: Filename to use as base
        """
        if not args.strip():
            print("\nError: Please specify a filename")
            print("Usage: /set-base <filename>")
            print(f"\nAvailable files: {list(self.extracted_tables.keys())}")
            return

        filename = args.strip()

        if filename not in self.extracted_tables:
            print(f"\nError: Table not extracted for: {filename}")
            print("\nUse '/extract' to extract tables first")
            return

        self.base_file = filename
        print(f"\n✓ Base file set to: {filename}")
        print(f"  Items: {len(self.extracted_tables[filename])}")
        print("\nNow use '/compare <file1> <file2> ...' to compare with other files")

    def cmd_compare_tables(self, args: str):
        """Compare base file with other files.

        Args:
            args: Space-separated list of filenames to compare
        """
        if not self.base_file:
            print("\nError: No base file set")
            print("Use '/set-base <filename>' first")
            return

        if not args.strip():
            # Compare with all other extracted tables
            comparison_files = [f for f in self.extracted_tables.keys() if f != self.base_file]
        else:
            # Compare with specified files
            comparison_files = args.strip().split()

        if not comparison_files:
            print("\nError: No comparison files specified")
            return

        # Validate all files are extracted
        for filename in comparison_files:
            if filename not in self.extracted_tables:
                print(f"\nError: Table not extracted for: {filename}")
                print("\nUse '/extract' to extract tables first")
                return

        print("\n" + "="*60)
        print("SEMANTIC TABLE COMPARISON")
        print("="*60)

        print(f"\nBase file: {self.base_file}")
        print(f"  Items: {len(self.extracted_tables[self.base_file])}")
        print(f"\nComparison files: {len(comparison_files)}")
        for f in comparison_files:
            print(f"  - {f} ({len(self.extracted_tables[f])} items)")

        # Ensure chat is started
        if not self.gemini_client.chat:
            print("\nStarting chat session for semantic matching...")
            self.gemini_client.start_chat()

        # Perform semantic matching
        base_df = self.extracted_tables[self.base_file]
        base_items = base_df['item'].tolist()

        match_results = {}

        for comp_file in comparison_files:
            print(f"\n{'='*60}")
            print(f"Comparing with: {comp_file}")
            print(f"{'='*60}")

            comp_df = self.extracted_tables[comp_file]
            comp_items = comp_df['item'].tolist()

            # Batch match
            results = self.semantic_matcher.batch_match(base_items, comp_items)
            match_results[comp_file] = results

            # Show statistics
            stats = self.semantic_matcher.get_match_statistics(results)
            print(f"\nMatch Statistics for {comp_file}:")
            print(f"  Total items: {stats['total_items']}")
            print(f"  Matched: {stats['matched']} ({stats['match_rate']:.1f}%)")
            print(f"  High confidence: {stats['high_confidence']}")
            print(f"  Medium confidence: {stats['medium_confidence']}")
            print(f"  Low confidence: {stats['low_confidence']}")
            print(f"  No match: {stats['no_match']}")

        # Build comparison table
        print(f"\n{'='*60}")
        print("BUILDING COMPARISON TABLE")
        print(f"{'='*60}\n")

        comparison_data = {f: self.extracted_tables[f] for f in comparison_files}
        comparison_df = self.comparison_generator.build_comparison_table(
            base_df, comparison_data, match_results
        )

        # Add statistics
        comparison_df = self.comparison_generator.add_statistics_row(comparison_df)

        # Store for export
        self.comparison_files = comparison_files
        self.last_comparison = comparison_df

        # Display summary
        summary = self.comparison_generator.generate_summary_report(
            base_df, comparison_data, match_results
        )
        print(summary)

        print("\n✓ Comparison complete!")
        print("Use '/export-comparison <filename>' to export results")

    def cmd_export_comparison(self, args: str):
        """Export comparison table to file.

        Args:
            args: Output filename (with .xlsx, .csv, or .md extension)
        """
        if not hasattr(self, 'last_comparison') or self.last_comparison is None:
            print("\nError: No comparison results to export")
            print("Run '/compare' first")
            return

        # Determine output filename
        if args.strip():
            filename = args.strip()
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.xlsx"

        output_path = Path('exports') / filename

        # Determine format from extension
        suffix = output_path.suffix.lower()

        if suffix == '.xlsx':
            success = self.comparison_generator.export_excel(
                self.last_comparison, output_path, add_formatting=True
            )
        elif suffix == '.csv':
            success = self.comparison_generator.export_csv(
                self.last_comparison, output_path
            )
        elif suffix == '.md':
            success = self.comparison_generator.export_markdown(
                self.last_comparison, output_path
            )
        else:
            print(f"\nError: Unsupported format: {suffix}")
            print("Supported formats: .xlsx, .csv, .md")
            return

        if success:
            print(f"\n✓ Comparison exported successfully!")
            print(f"  Location: {output_path}")
        else:
            print(f"\n✗ Export failed")
