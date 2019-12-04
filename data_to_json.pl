use strict;
use warnings;
use File::Slurp;
use Jq;

my @files = glob("temp*");

foreach my $file (@files){
	print $file ."\n";
	my @file_content = read_file($file);
	my @data = jq -r '.[]', @file_content;
	open(my $fh, '>', 'temp_res.json');
	print $fh @data;
	close $fh;
	print "Done...\n";
}
